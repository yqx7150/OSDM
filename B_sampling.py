import math
import os
from copy import deepcopy
from random import betavariate
import sys
# region import
sys.path.append('..')
import functools
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse

import torch
import numpy as np
import abc
from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim, \
    mean_squared_error as compare_mse
import odl
import glob
import pydicom
from cv2 import imwrite, resize
from func_test import WriteInfo, WriteInfo_zl
from scipy.io import loadmat, savemat
from radon_utils import (create_sinogram, bp, filter_op,
                         fbp, reade_ima, write_img, sinogram_2c_to_img,
                         padding_img, unpadding_img, indicate)
from time import sleep

import odl

# endregion
Fan_angle_partition = odl.uniform_partition(0, 2 * np.pi, 720)
Fan_detector_partition = odl.uniform_partition(-180, 180, 720)
# Fan_geometry = odl.tomo.FanBeamGeometry(Fan_angle_partition, Fan_detector_partition,
#                             src_radius=500, det_radius=500)
Fan_geometry = odl.tomo.Parallel2dGeometry(Fan_angle_partition, Fan_detector_partition)
Fan_reco_space = odl.uniform_discr(min_pt=[-128, -128], max_pt=[128, 128], shape=[512, 512], dtype='float32')
Fan_ray_trafo = odl.tomo.RayTransform(Fan_reco_space, Fan_geometry)
Fan_FBP = odl.tomo.fbp_op(Fan_ray_trafo)
Fan_filter = odl.tomo.fbp_filter_op(Fan_ray_trafo)

_CORRECTORS = {}
_PREDICTORS = {}

# region function
def set_predict(num):
    if num == 0:
        return 'None'
    elif num == 1:
        return 'EulerMaruyamaPredictor'
    elif num == 2:
        return 'ReverseDiffusionPredictor'

def set_correct(num):
    if num == 0:
        return 'None'
    elif num == 1:
        return 'LangevinCorrector'
    elif num == 2:
        return 'AnnealedLangevinDynamics'

def padding_img(img):
    b, w, h = img.shape
    h1 = 768
    tmp = np.zeros([b, h1, h1])
    x_start = int((h1 - w) // 2)
    y_start = int((h1 - h) // 2)
    tmp[:, x_start:x_start + w, y_start:y_start + h] = img
    return tmp

def unpadding_img(img):
    b, w, h = img.shape[0], 720, 720
    h1 = 768
    tmp = np.zeros([b, h1, h1])
    x_start = int((h1 - w) // 2)
    y_start = int((h1 - h) // 2)
    return img[:, x_start:x_start + w, y_start:y_start + h]

def init_ct_op(img, r):
    batch = img.shape[0]
    sinogram = np.zeros([batch, 720, 720])
    sparse_sinogram = np.zeros([batch, 720, 720])
    ori_img = np.zeros_like(img)
    sinogram_max = np.zeros([batch, 1])
    for i in range(batch):
        sinogram[i, ...] = Fan_ray_trafo(img[i, ...]).data
        ori_img[i, ...] = Fan_FBP(sinogram[i, ...]).data
        sinogram_max[i, 0] = sinogram[i, ...].max()
        # sinogram[i,...] /= sinogram_max[i,0]
        t = np.copy(sinogram[i, ::r, :])
        sparse_sinogram[i, ...] = resize(t, [720, 720])

    return ori_img, sparse_sinogram.astype(np.float32), sinogram.astype(np.float32), sinogram_max

def init_ct_op_cube(sinogram, r, cube2DSize):
    batch = sinogram.shape[0]
    sparse_sinogram = np.zeros_like(sinogram)
    ori_img = np.zeros([batch,cube2DSize,cube2DSize],dtype=np.float32)
    sinogram_max = np.zeros([batch, 1])
    for i in range(batch):
        ori_img[i, ...] = Fan_FBP(sinogram[i, ...]).data
        sinogram_max[i, 0] = sinogram[i, ...].max()
        # sinogram[i,...] /= sinogram_max[i,0]
        t = np.copy(sinogram[i, ::r, :])
        sparse_sinogram[i, ...] = resize(t, sinogram.shape)

    return ori_img, sparse_sinogram.astype(np.float32), sinogram.astype(np.float32), sinogram_max
# endregion

# region init

def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):

    sampler_name = config.sampling.method  # pc
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == 'ode':
        sampling_fn = get_ode_sampler(sde=sde,
                                      shape=shape,
                                      inverse_scaler=inverse_scaler,
                                      denoise=config.sampling.noise_removal,
                                      eps=eps,
                                      device=config.device)
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == 'pc':
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(sde=sde,
                                     shape=shape,
                                     predictor=predictor,
                                     corrector=corrector,
                                     inverse_scaler=inverse_scaler,
                                     snr=config.sampling.snr,
                                     n_steps=config.sampling.n_steps_each,
                                     probability_flow=config.sampling.probability_flow,
                                     continuous=config.training.continuous,
                                     denoise=config.sampling.noise_removal,
                                     eps=eps,
                                     device=config.device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


# ===================================================================== ReverseDiffusionPredictor
@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


# =====================================================================

@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        assert not probability_flow, "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
        score = self.score_fn(x, t)
        x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
        std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update_fn(self, x, t):
        if isinstance(self.sde, sde_lib.VESDE):
            return self.vesde_update_fn(x, t)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x


# ================================================================================================== LangevinCorrector
@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


# ==================================================================================================

@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x


# ========================================================================================================

def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):  # gb A改坐标
    # def shared_corrector_update_fn(x1,x2,x3,x_mean,t,sde, model, corrector, continuous, snr, n_steps): # ph
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    # return corrector_obj.update_fn(x1, x2, x3, x_mean, t) # ph 代码
    return corrector_obj.update_fn(x, t)  # gb 代码 A改坐标

# endregion

# ========================================================================================================

def get_pc_sampler(sde, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
    # region init
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)
    # endregion
    def pc_sampler(img_model, check_num, predict, correct, zl_args):
        file_name = ['0319', '0160', '0001', '0354', '0405', '0292', '0462', '0137', '0521', '0062', '0248', '0361']

        # R = np.array([6,8,12])
        # R = np.array([12,8,6,4])

        # cut0 = zl_args.cut0
        cut0 = 10  # 0~256
        zl_args.pic = 0

        # R = np.array([zl_args.r])
        # R     4   6   8  12
        # view  180 120 90 60
        # R = np.array([4, 6, 8, 12])
        # R = np.array([4, 6, 8, 12])
        R = np.array([4])
        views = 720 // R
        ksize = [16, 16]

        useSvd = True
        useDebug = False
        guiyihua = True
        showimg = False

        for rid, r in enumerate(R):
            view = views[rid]

            # [9, 256, 256] into net
            imgSize = 768
            cube2DSize = 64

            cubeChannel = 9
            cubeSize = 256

            cubeNum = int((imgSize / cube2DSize)*(imgSize / cube2DSize))
            # 144

            img = np.load('./Test_CT/batch_img.npy')
            img = img[zl_args.pic:zl_args.pic + 1, :, :]
            # [1, 512, 512]

            # use ori img
            # img = np.load('../train_img/1.npy')
            # img = img[None,:,:]


            ori_img_Big, sparse_sinogram_Big, sinogram_Big, sinogram_max_Big = init_ct_op(img, r)
            # (1, 512, 512) (1, 720, 720) (1, 720, 720) (1, 1)

            # plt.imshow(sparse_sinogram_Big[0,:,:], cmap=plt.get_cmap('gray'))
            # plt.show()
            #
            # plt.imshow(sinogram_Big[0, :, :], cmap=plt.get_cmap('gray'))
            # plt.show()

            # region 创建多个小块cube的方法
            def createCube(img, imgSize, cube2DSize):
                cubeNum = int((imgSize / cube2DSize) * (imgSize / cube2DSize))
                cubeLineNum = int(imgSize / cube2DSize)
                cutList_ = []
                for i in range(cubeNum):
                    startX = 0 + (i // cubeLineNum) * cube2DSize
                    startY = 0 + i % cubeLineNum * cube2DSize
                    # print(f"{i}  {startX}  {startY}")
                    temp = img[startX:startX + cube2DSize, startY:startY + cube2DSize]
                    cutList_.insert(i, temp)
                    # plt.imshow(temp, cmap=plt.get_cmap('gray'))
                    # plt.savefig(f'img_{i}.png')
                return cutList_

            def integrateCube(cubeList, imgSize, cube2DSize):
                cubeNum = int((imgSize / cube2DSize) * (imgSize / cube2DSize))
                cubeLineNum = int(imgSize / cube2DSize)

                img_big_ = np.zeros((imgSize, imgSize), dtype=np.float32)
                for i in range(cubeNum):
                    startX = 0 + (i // cubeLineNum) * cube2DSize
                    startY = 0 + i % cubeLineNum * cube2DSize
                    # print(f"rebuild {i}  {startX}  {startY}")
                    temp = cubeList[i]
                    img_big_[startX:startX + cube2DSize, startY:startY + cube2DSize] = temp[:, :]

                return img_big_

            def createCubeAll(sparse_sinogram_Big, sinogram_Big, imgSize, cube2DSize):
                sparse_sinogram_Big = sparse_sinogram_Big.squeeze()
                sinogram_Big = sinogram_Big.squeeze()
                return createCube(sparse_sinogram_Big, imgSize, cube2DSize), \
                       createCube(sinogram_Big, imgSize, cube2DSize)

            # endregion

            with torch.no_grad():
                batch_size = ori_img_Big.shape[0]
                # 1

                sinogram_Big = padding_img(sinogram_Big).astype(np.float32)
                sparse_sinogram_Big = padding_img(sparse_sinogram_Big).astype(np.float32)

                # region hankelTool
                def toNumpy(tensor):
                    return tensor.cpu().numpy()
                def toTensor(array):
                    return torch.from_numpy(array).cuda()
                def im2row(im, winSize):
                    size = (im).shape
                    out = np.zeros(
                        ((size[0] - winSize[0] + 1) * (size[1] - winSize[1] + 1), winSize[0] * winSize[1], size[2]),
                        dtype=np.float32)
                    count = -1
                    for y in range(winSize[1]):
                        for x in range(winSize[0]):
                            count = count + 1
                            temp1 = im[x:(size[0] - winSize[0] + x + 1), y:(size[1] - winSize[1] + y + 1), :]
                            temp2 = np.reshape(temp1,
                                               [(size[0] - winSize[0] + 1) * (size[1] - winSize[1] + 1), 1, size[2]],
                                               order='F')
                            # out[:, count, :] = np.squeeze(temp2)  # MATLAB reshape
                            out[:, count, 0] = np.squeeze(temp2)  # MATLAB reshape

                    return out

                def svd_zl(input, cutSize=64):
                    if useSvd == False:
                        return np.array(input, dtype=np.float32)
                    # SVD ===============================================
                    svd_input = torch.tensor(input, dtype=torch.float32)
                    U, S, V = torch.svd(svd_input)
                    S = torch.diag(S)

                    U = np.array(U, dtype=np.float32)  # (579121, 64)
                    S = np.array(S, dtype=np.float32)  # (64, 64)
                    V = np.array(V, dtype=np.float32)  # (64, 64)
                    # zero jie duan
                    uu = U[:, 0:math.floor(cutSize)]  # (63001, 64)
                    ss = S[0:math.floor(cutSize),
                         0:math.floor(cutSize)]  # (64, 64)
                    vv = V[:, 0:math.floor(cutSize)]  # (288,64)

                    A_svd = np.dot(np.dot(uu, ss), vv.T)
                    # get A_svd -- 61952*512  cutSize=512 ======================================
                    return A_svd

                def row2im(mtx, size_data, winSize):
                    size_mtx = mtx.shape  # (63001, 36, 8)
                    sx = size_data[0]  # 256
                    sy = size_data[1]  # 256
                    sz = size_mtx[2]  # 8

                    res = np.zeros((sx, sy, sz), dtype=np.float32)
                    W = np.zeros((sx, sy, sz), dtype=np.float32)
                    out = np.zeros((sx, sy, sz), dtype=np.float32)
                    count = -1

                    for y in range(winSize[1]):
                        for x in range(winSize[0]):
                            count = count + 1
                            res[x: sx - winSize[0] + x + 1, y: sy - winSize[1] + y + 1, :] = res[
                                                                                             x: sx - winSize[0] + x + 1,
                                                                                             y: sy - winSize[1] + y + 1,
                                                                                             :] + np.reshape(
                                np.squeeze(mtx[:, count, :]), [sx - winSize[0] + 1, sy - winSize[1] + 1, sz], order='F')
                            W[x: sx - winSize[0] + x + 1, y: sy - winSize[1] + y + 1, :] = W[x: sx - winSize[0] + x + 1,
                                                                                           y: sy - winSize[1] + y + 1,
                                                                                           :] + 1

                    out = np.multiply(res, 1. / W)
                    return out

                def hankel_zl(input, ksize):
                    '''

                    Args:
                        input: [x*y*c] c is channel
                        ksize:

                    Returns:

                    '''
                    hankel = im2row(input, ksize)
                    size_temp = hankel.shape
                    A = np.reshape(hankel, [size_temp[0], size_temp[1] * size_temp[2]], order='F')
                    return A, size_temp

                def hankel_ni_zl(input, ksize, size_temp, size_last=None):
                    '''

                    Args:
                        input: [9999,99]
                        ksize: [8,8]
                        size_temp:
                        size_last: for example, [768, 768, 1]

                    Returns:

                    '''
                    if size_last is None:
                        size_last = [768, 768, 1]
                    input = np.reshape(input, [size_temp[0], size_temp[1], size_temp[2]], order='F')
                    B = row2im(input, size_last, ksize)
                    return B


                def unfoldTo2D(thing_3D, index):
                    # A's size -- 9*256*256
                    '''
                    useage:
                        A_new = unfoldTo2D(A_complex, 1)
                        A_new = svd_zl(A_new, cut1)
                        A_new = pileTo3D(A_new, 1)
                    Args:
                        thing_3D:
                        index: 0,1,2
                    Returns:
                    '''

                    size_3D = thing_3D.shape
                    thing_2D = np.zeros((size_3D[0] * size_3D[1], size_3D[2]), dtype=np.float32)
                    if index == 0:
                        for i in range(size_3D[0]):
                            thing_2D[size_3D[1] * i:size_3D[1] * (i + 1), :] = thing_3D[i, :, :]
                    elif index == 1:
                        for i in range(size_3D[1]):
                            thing_2D[size_3D[0] * i:size_3D[0] * (i + 1), :] = thing_3D[:, i, :]
                    elif index == 2:
                        for i in range(size_3D[2]):
                            thing_2D[size_3D[0] * i:size_3D[0] * (i + 1), :] = thing_3D[:, :, i]
                    else:
                        return -996

                    return thing_2D

                def pileTo3D(thing_2D, index, size_3D=(121, 512, 512)):
                    # B's size -- 61952*512
                    size_2D = thing_2D.shape
                    thing_3D = np.zeros((size_3D[0], size_3D[1], size_3D[2]), dtype=np.float32)
                    if index == 0:
                        for i in range(size_3D[0]):
                            thing_3D[i, :, :] = thing_2D[size_3D[1] * i:size_3D[1] * (i + 1), :]
                    elif index == 1:
                        for i in range(size_3D[1]):
                            thing_3D[:, i, :] = thing_2D[size_3D[0] * i:size_3D[0] * (i + 1), :]
                    elif index == 2:
                        for i in range(size_3D[2]):
                            thing_3D[:, :, i] = thing_2D[size_3D[0] * i:size_3D[0] * (i + 1), :]
                    else:
                        return -996

                    return thing_3D

                # endregion

                timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

                max_psnr = np.zeros(batch_size)
                max_ssim = np.zeros(batch_size)
                min_mse = 999 * np.ones(batch_size)
                rec_img_Big = np.zeros_like(ori_img_Big)
                best_img_Big = np.zeros_like(ori_img_Big)
                # (1, 768, 768)

                # sinogram_Big = torch.from_numpy(sinogram_Big).cuda()
                # [1,768,768]
                sparse_sinogram_List, sinogram_List = createCubeAll(sparse_sinogram_Big,sinogram_Big,imgSize,cube2DSize)
                # 遍历多个小块
                for cubeNO in range(cubeNum):
                    print(f"cubeNO is : {cubeNO}")
                    sparse_sinogram = sparse_sinogram_List[cubeNO]
                    sinogram = sinogram_List[cubeNO]
                    # [64,64]
                    # 历史遗留问题，是后面的格式要求
                    sinogram = sinogram[None, ...]
                    x0 = sparse_sinogram

                    x0max = np.max(abs(x0))
                    if guiyihua:
                        sinoMax = np.max(abs(sinogram))
                        sinogram = sinogram / sinoMax

                        x0max =  np.max(abs(x0))
                        x0 = x0 / x0max

                        # print(
                        #     f"sinoMax: {sinoMax} sinoMaxNow : {np.max(sinogram)} x0max: f{x0max} x0MaxNow: {np.max(x0)}")

                    # print(f"sinoMaxNow : {np.max(sinogram)} x0MaxNow: {np.max(x0)}")

                    sinogram = toTensor(sinogram)

                    # region recon

                    # region PC  sde.N
                    for i in range(997):
                        if i % 50 == 0 :
                            print(f"r: {r} cube: {cubeNO} steps : {i}")
                        t = timesteps[i]
                        vec_t = torch.ones(x0[None,...].shape[0], device=t.device) * t

                        # [64, 64]
                        x0 = x0[..., None]
                        # [64,64,1]
                        A,size_temp = hankel_zl(x0,ksize)
                        # [2401,256]
                        x0 = pileTo3D(A,0,[cubeChannel,cubeSize,cubeSize])
                        x0 = toTensor(x0[None,...])
                        # [1,9,256,256]

                        if useDebug:
                            print("=====================================")
                            print(f'first into predict : {x0.shape}')

                        # =========================================================PREDICT
                        # print(f"before predict result max : {np.max(toNumpy(x0))} result min: {np.min(toNumpy(x0))}")

                        x01, x0 = predictor_update_fn(x0, vec_t, model=img_model)

                        # print(f"after predict result max : {np.max(toNumpy(x0))} result min: {np.min(toNumpy(x0))}")
                        # [1,9,256,256]
                        if useDebug:
                            print(f'after predict : {x0.shape}')

                        x0 = unfoldTo2D(toNumpy(x0).squeeze(),0)
                        if useDebug:
                            print(f'after unfold : {x0.shape}')
                        # [2304,256]
                        A_fix = A[cubeChannel * cubeSize:, :]
                        x0 = np.concatenate((x0, A_fix), 0)
                        # [2401,256] 已通过无PC的相等性校验，A与x0
                        x0 = svd_zl(x0, cut0)
                        if useDebug:
                            print(f'after svd: {x0.shape}')

                        # 逆hankel
                        x0 = hankel_ni_zl(x0, ksize, size_temp, [cube2DSize, cube2DSize, 1])
                        if useDebug:
                            print(f'after hankel_ni: {x0.shape}')
                        #  [64,64,1]

                        ## DC
                        x0 = torch.from_numpy(x0).cuda()
                        # [64,64,1] / [1, 64, 64]
                        x0[::r, :, 0] = sinogram[0, ::r, :]  # [:,::,65]
                        x0 = x0.cpu().numpy()
                        # [64,64,1]
                        if useDebug:
                            print(f'after svd and ni_hankel : {x0.shape}')

                        # cube2D to cube3D
                        A, size_temp = hankel_zl(x0, ksize)
                        # [2401,256]
                        x0 = pileTo3D(A, 0, [cubeChannel, cubeSize, cubeSize])
                        x0 = toTensor(x0[None, ...])
                        # [1,9,256,256]
                        if useDebug:
                            print(f'before into predict : {x0.shape}')

                        # =========================================================CORRECT
                        # print(f"before correct result max : {np.max(toNumpy(x0))} result min: {np.min(toNumpy(x0))}")

                        # x01, x0 = corrector_update_fn(x0, vec_t, model=img_model)
                        x01, x0 = corrector_update_fn(x0, vec_t, model=img_model)

                        # print(f"after correct result max : {np.max(toNumpy(x0))} result min: {np.min(toNumpy(x0))}")

                        # [1,9,256,256]
                        if useDebug:
                            print(f'after correct : {x0.shape}')

                        x0 = unfoldTo2D(toNumpy(x0).squeeze(), 0)
                        # [2304,256]
                        if useDebug:
                            print(f'after unfold : {x0.shape}')
                        A_fix = A[cubeChannel * cubeSize:, :]
                        x0 = np.concatenate((x0, A_fix), 0)
                        # [2401,256] 已通过无PC的相等性校验，A与x0
                        x0 = svd_zl(x0, cut0)
                        if useDebug:
                            print(f'after svd: {x0.shape}')

                        # 逆hankel
                        x0 = hankel_ni_zl(x0, ksize, size_temp, [cube2DSize, cube2DSize, 1])
                        if useDebug:
                            print(f'after hankel_ni: {x0.shape}')
                        #  [64,64,1]

                        ## DC


                        x0 = torch.from_numpy(x0).cuda()
                        # [64,64,1] / [1, 64, 64]
                        x0[::r, :, 0] = sinogram[0, ::r, :]  # [:,::,65]
                        x0 = x0.cpu().numpy()
                        # [64,64,1]
                        if useDebug:
                            print(f'after svd and ni_hankel : {x0.shape}')

                        # plt.imshow(x0, cmap=plt.get_cmap('gray'))
                        # plt.show()


                    # endregion PC

                    # 还原
                    if guiyihua:
                        x0 = x0 / np.max(x0)
                        print(f"x0MaxAfterGYH:  {np.max(x0)}")
                        x0 = x0 * sinoMax
                        print(f"x0MaxAfterRecover:  {np.max(x0)}")

                    # 超级修复？
                    x0 =toTensor(x0)
                    fucker = sinogram
                    if guiyihua:
                        fucker = sinogram * sinoMax
                    # [64,64,1] / [1, 64, 64]
                    #x0[1:10:, :, 0] = fucker[0, 1:10:, :]  #
                    x0[::r, :, 0] = fucker[0,::r, :]  #
                    x0 = x0.cpu().numpy()
                    x0 = np.where(x0 > 0, x0, 0) # 小于零变零

                    # 64 64 1
                    x0 = x0.squeeze()
                    sparse_sinogram_List[cubeNO] = deepcopy(x0)
                    sparse_sinogram_Big = integrateCube(sparse_sinogram_List,imgSize,cube2DSize)

                    plt.imshow(sparse_sinogram_Big, cmap=plt.get_cmap('gray'))
                    plt.savefig(f'./result_zl/img{cubeNO}.png')
                    if showimg:
                        plt.show()


                    print(f"result max : {np.max(sparse_sinogram_Big)} result min: {np.min(sparse_sinogram_Big)}")

                    # region PSNR
                    i = cubeNO

                    tmp = np.squeeze(sparse_sinogram_Big)
                    tmp = tmp[None, ...]
                    tmp = unpadding_img(tmp)
                    for coil in range(batch_size):
                        rec_img_Big[coil, ...] = Fan_FBP(tmp[coil, ...] * sinogram_max_Big[coil, ...]).data

                    write_img(tmp[0, ...], 'rec_sinogram.png')
                    write_img(rec_img_Big[0, ...])


                    psnr0, ssim0, mse0 = indicate(rec_img_Big, ori_img_Big)

                    c = max_psnr < psnr0
                    if np.sum(c) > 0.01:
                        max_psnr = max_psnr * (1 - c) + psnr0 * c
                        max_ssim = max_ssim * (1 - c) + ssim0 * c
                        min_mse = min_mse * (1 - c) + mse0 * c
                        min_mse = min_mse * (1 - c) + mse0 * c
                        c = c[..., None, None]
                        best_img_Big = best_img_Big * (1 - c) + rec_img_Big * c

                    print(
                        f"Step: {i}   Views:{view}  PSNR:{np.round(psnr0[:4], 3)}  MSE:{np.round(1000 * mse0[:4], 3)}   R: {r}   pic: {zl_args.pic}   cut0: {zl_args.cut0}    cubeNO: {cubeNO}")

                    path_dir = f"./result_zl/{r}"
                    path_filename = f"P_{zl_args.pic}_R_{r}_cut_{zl_args.cut0}"
                    path_all_step = f"{path_dir}/{path_filename}_all.csv"
                    path_best_step = f"{path_dir}/{path_filename}.csv"
                    path_best_step_img = f"{path_dir}/{path_filename}.png"
                    path_best_npy = f"{path_dir}/{path_filename}.npy"
                    path_best_mat = f"{path_dir}/{path_filename}.mat"

                    # delete all step when get a new start
                    if i == 0:
                        t = 0
                        if os.path.exists(path_best_step):
                            os.remove(path_best_step)
                            t = t + 1
                        if os.path.exists(path_all_step):
                            os.remove(path_all_step)
                            t = t + 1
                        if t == 2:
                            print("Delete old files succeed ... Now it is a New start....")

                    # save all step
                    if i >= 0:
                        WriteInfo_zl(path_all_step,
                                     图片=zl_args.pic, 稀疏角=view, svd=zl_args.cut0,
                                     PSNR=psnr0[0], SSIM=ssim0[0], MSE=mse0[0],
                                     step=i, Check_num=check_num, cubeNO=cubeNO)
                    # save best step
                    if i == 998:  # i==998
                        WriteInfo_zl(path_best_step,
                                     图片=zl_args.pic, 稀疏角=view, svd=zl_args.cut0,
                                     PSNR=max_psnr[0], SSIM=max_ssim[0], MSE=min_mse[0],
                                     step=i, Check_num=check_num, cubeNO=cubeNO)
                        np.save(path_best_npy, best_img_Big)
                        savemat(path_best_mat, mdict={'data': best_img_Big})

                        plt.imshow(best_img_Big[0, :, :], cmap=plt.get_cmap('gray'))
                        plt.savefig(path_best_step_img)

                        # np.save(f'./result_zl/{zl_args.pic}/result_{str}.npy', best_img)

                    # endregion

                    # endregion
                print("MAX:  PSNR:{} MSE:{}".format(np.round(max_psnr[:4], 3), np.round(1000 * min_mse[:4], 3)))


    return pc_sampler





def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
    """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

    def denoise_update_fn(model, x):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps)
        return x

    def drift_fn(model, x, t):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(model, z=None):
        """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x)

            x = inverse_scaler(x)
            return x, nfe

    return ode_sampler
