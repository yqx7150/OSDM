# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc
from operator_fza import forward,backward

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils
#from skimage.measure import compare_psnr,compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import matplotlib.pyplot as plt
from fza_mask import fza
from fza_mask_forward import fza_mask_forward
from TV_norm import TVnorm
from tvdenoise import tvdenoise
import scipy.io as io
from lmafit_mc_adp_gpu import lmafit_mc_adp

_CORRECTORS = {}
_PREDICTORS = {}


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
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
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


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x1,x2,x3,x_mean, t):
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
      grad1 = score_fn(x1, t) # 5 
      grad2 = score_fn(x2, t) # 5 
      grad3 = score_fn(x3, t) # 5 
      
      noise1 = torch.randn_like(x1) # 4 
      noise2 = torch.randn_like(x2) # 4
      noise3 = torch.randn_like(x3) # 4
      
      grad_norm1 = torch.norm(grad1.reshape(grad1.shape[0], -1), dim=-1).mean()
      noise_norm1 = torch.norm(noise1.reshape(noise1.shape[0], -1), dim=-1).mean()
      grad_norm2 = torch.norm(grad2.reshape(grad2.shape[0], -1), dim=-1).mean()
      noise_norm2 = torch.norm(noise2.reshape(noise2.shape[0], -1), dim=-1).mean()      
      grad_norm3 = torch.norm(grad3.reshape(grad3.shape[0], -1), dim=-1).mean()
      noise_norm3 = torch.norm(noise3.reshape(noise3.shape[0], -1), dim=-1).mean()            
      
      grad_norm =(grad_norm1+grad_norm2+grad_norm3)/3.0
      noise_norm = (noise_norm1+noise_norm2+noise_norm3)/3.0
      
      step_size =  (2 * alpha)*((target_snr * noise_norm / grad_norm) ** 2 ) # 6 
   
      x_mean = x_mean + step_size[:, None, None, None] * (grad1+grad2+grad3)/3.0 # 7
      
      x1 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise1 # 7
      x2 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise2 # 7
      x3 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise3 # 7
    return x1,x2,x3, x_mean


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


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x1,x2,x3,x_mean, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(x1,x2,x3,x_mean, t)
################################################################################################################

def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def im2row(im,winSize):
  size = (im).shape                    #(256,256,3)
  out = torch.zeros(((size[0]-winSize[0]+1)*(size[1]-winSize[1]+1),winSize[0]*winSize[1],size[2]),dtype=torch.float64).cuda() #(62001,64,3)
  count = -1
  for y in range(winSize[1]):
    for x in range(winSize[0]):
      count = count + 1                 
      temp1 = im[x:(size[0]-winSize[0]+x+1),y:(size[1]-winSize[1]+y+1),:]       #im[0:57,0:57,:]    窗口是57×57大小
    #   temp3 = np.reshape(temp1.cpu(),[(size[0]-winSize[0]+1)*(size[1]-winSize[1]+1),1,size[2]],order = 'F')

      temp2 = reshape_fortran(temp1,[(size[0]-winSize[0]+1)*(size[1]-winSize[1]+1),1,size[2]])      #reshape_fortran(temp1,[57*57,1,3])
      
      out[:,count,:] = temp2.squeeze() # MATLAB reshape          
		
  return out

def row2im(mtx,size_data,winSize):
    size_mtx = mtx.shape #(62001, 64, 3)
    sx = size_data[0] # 256
    sy = size_data[1] # 256
    sz = size_mtx[2] # 3
    
    res = torch.zeros((sx,sy,sz),dtype=torch.float64).cuda()      #(256,256,3)
    W = torch.zeros((sx,sy,sz),dtype=torch.float64).cuda()        #(256,256,3)
    out = torch.zeros((sx,sy,sz),dtype=torch.float64).cuda()      #(256,256,3)
    count = -1

    # aaaa = np.reshape(np.squeeze(mtx[:,count,:]).cpu(),[sx-winSize[0]+1,sy-winSize[1]+1,sz],order = 'F')
    # bbbb = reshape_fortran((mtx[:,count,:]).squeeze(),[sx-winSize[0]+1,sy-winSize[1]+1,sz])

    # print( '111111111',(aaaa == bbbb.cpu()).all())
    # assert 0
    
    for y in range(winSize[1]):
        for x in range(winSize[0]):
            count = count + 1
            res[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] = res[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] + reshape_fortran((mtx[:,count,:]).squeeze(),[sx-winSize[0]+1,sy-winSize[1]+1,sz])  
            W[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] = W[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] + 1
            

    # out = np.multiply(res,1./W)
    out = torch.mul(res,1./W)
    return out
###############################################################################################################

def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions 第一个是函数，其他的是参数
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

  def pc_sampler(model,data,mask,data_ob):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      x_mean = sde.prior_sampling((1,3,256,256)).to(device)    #(1,3,256,256)
      x_mean_hankel = x_mean.squeeze(0).permute(1,2,0)          #(256,256,3)
     # x_mean_hankel = x_mean_hankel-x_mean_hankel.min()
#####################################################################################    
      #x1=x_mean
      #x2=x_mean
      #x3=x_mean
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)    #(1,1e-5,1000)
      psnr_max=0
      ssim_max=0
      psnr_n=[]
      ssim_n=[]
###################################################################
      Nfir=8
      Nimg=256
      size_data = [Nimg,Nimg,3]
      ksize = [Nfir,Nfir]
##############################图像转化为hankel######################################

      hankel = im2row(x_mean_hankel,ksize)           #(62001,64,3)       图像转化成hankel矩阵
  
      size_temp = hankel.shape    #(62001,64,3) 
      A = reshape_fortran(hankel,[size_temp[0],size_temp[1]*size_temp[2]])    #(62001,192)

##############################将hankel矩阵转为tensor################################################
      ans_1 = torch.zeros((322,192,192),dtype=torch.float64)
      for i in range(322):
        cut = A[i*192:(i+1)*192,:]
        ans_1[i,:,:] = cut          #(322,192,192)

      x_input=ans_1.to(torch.float32).cuda().unsqueeze(0)     #tensor:(1,322,192,192)
      x_mean_patch = x_input
      
        
#############################################################################################
      #for i in range(sde.N):
      for n in range(sde.N):
        t = timesteps[n].cuda()
        vec_t = torch.ones(x_input.shape[0], device=t.device) * t                                
     
###########################################预测器更新#######################
        x, x_mean_patch = predictor_update_fn(x_mean_patch, vec_t, model=model)#预测器更新       #################################修改x_mean

        #x: A PyTorch tensor of the next state.     
        #x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
      
##############################################tensor转化为hankel######################################
        x_mean_patch = x_mean_patch.clone().detach().squeeze(0)       #tensor:(322,192,192)
        A_new=torch.zeros((322*192,192),dtype=torch.float64)
        for i in range(322): 
          A_new[192*i:192*(i+1),:]=x_mean_patch[i,:,:]
        A_no=A[322*192:,:]
        P_output=torch.cat((A_new.cuda(),A_no),0) # (62001, 192)          #将训练好的hankel矩阵和没训练的hankel矩阵部分进行拼接
##############################################hankel转化为图像#############################################
        P_output = reshape_fortran(P_output,[P_output.shape[0],int(P_output.shape[1]/3),3])   #（62001,64,3） 
          
          
        #print(x_mean_patch.min(),x_mean_patch.max())
           
        kcomplex_h = row2im(P_output, size_data,ksize )         #tensor(256,256,3)
 
       # kcomplex_h = (kcomplex_h-kcomplex_h.min())/(kcomplex_h.max()-kcomplex_h.min())
         
##############################################################################################################
        '''
        data=data[0,:,:,:].cpu().numpy().transpose(1,2,0)
        data=np.expand_dims(data,axis=0)
        data = torch.from_numpy(data).permute(0, 3, 1, 2).cuda()
        '''
        data_ob_first=data_ob.permute(2,0,1).unsqueeze(0)       #(1,3,256,256)           #####################################
        '''
        plt.imshow(data[0,1,:,:].cpu().numpy(),cmap='gray')
        plt.show()
        '''

        masked_data_mean, std = sde.marginal_prob(data_ob_first, vec_t) #std = self.sigma_min * (self.sigma_max / self.sigma_min) ** vec_t      masked_data_mean = data_ob_first

        forward_noise_0,forward_noise_1,forward_noise_2=fza_mask_forward(torch.randn_like(x_mean))      #(256,256)
        noise_result=torch.stack((forward_noise_0,forward_noise_1,forward_noise_2),dim=0).unsqueeze(0)   #tensor:(1,3,256,256)


        masked_data = masked_data_mean +  noise_result* std[:, None, None, None]      #tensor:(1,3,256,256)

        masked_data_mean = masked_data_mean.squeeze(0).permute(1,2,0)         #tensor:(256,256,3)
  
        masked_data = masked_data.squeeze(0).permute(1,2,0)                  #tensor:(256,256,3)

          
###########################################################################################################################################
        a=1
        ####################################x_mean保真################################################
          
        x_red=kcomplex_h[:,:,0]+a*backward(masked_data[:,:,0]-forward(kcomplex_h[:,:,0],mask),mask)    #注意backward（）中的cut_F_masked_data[j,:,:,0]是频域的
  
        #x_mean---x_mean_patch;   mask_data---cut_F_masked_data;  mask_data_mean---cut_F_masked_data_men;    mask---cut_mask;
 
        x_green=kcomplex_h[:,:,1]+a*backward(masked_data[:,:,1]-forward(kcomplex_h[:,:,1],mask),mask) 
        x_blue=kcomplex_h[:,:,2]+a*backward(masked_data[:,:,2]-forward(kcomplex_h[:,:,2],mask),mask)
        kcomplex_h=torch.stack((x_red,x_green,x_blue),dim=2)        #(256,256,3)
          
        ####################################x保真#####################################################
          
        x_red=kcomplex_h[:,:,0]+backward(masked_data_mean[:,:,0]-forward(kcomplex_h[:,:,0],mask),mask)     #注意backward（）中的cut_F_masked_data_mean[j,:,:,0]是频域的              
        x_green=kcomplex_h[:,:,1]+backward(masked_data_mean[:,:,1]-forward(kcomplex_h[:,:,1],mask),mask)
        x_blue=kcomplex_h[:,:,2]+backward(masked_data_mean[:,:,2]-forward(kcomplex_h[:,:,2],mask),mask)
        x=torch.stack((x_red,x_green,x_blue),dim=2)                  #(256,256,3)
        #x1,x2,x3=x,x,x
          
##################################################图像转为hankel########################################################################################
       # kcomplex_h = (kcomplex_h-kcomplex_h.min())/(kcomplex_h.max()-kcomplex_h.min())
        hankel = im2row(kcomplex_h,ksize)           #(62001,64,3)       转化成hankel矩阵
        size_temp = hankel.shape
        A = reshape_fortran(hankel,[size_temp[0],size_temp[1]*size_temp[2]])    #(62001,192)
          
        x_hankel = im2row(x,ksize)           #(62001,64,3)       转化成hankel矩阵
        x_size_temp = x_hankel.shape
        x_P_A = reshape_fortran(x_hankel,[x_size_temp[0],x_size_temp[1]*x_size_temp[2]])    #(62001,192)
          
#####################################################hankel转化为tensor####################################################################################
        ans_1 = torch.zeros((322,192,192),dtype=torch.float64)
        x_ans_1 = torch.zeros((322,192,192),dtype=torch.float64)
        for i in range(322):
          cut = A[i*192:(i+1)*192,:]
          ans_1[i,:,:] = cut          #(322,192,192)
          x_cut = x_P_A[i*192:(i+1)*192,:]
          x_ans_1[i,:,:] = x_cut      #(322,192,192)
        x_mean_patch = ans_1.to(torch.float32).cuda().unsqueeze(0)         #(1,322,192,192)
        x_patch = x_ans_1.to(torch.float32).cuda().unsqueeze(0)            #(1,322,192,192)
        x1,x2,x3 = x_patch,x_patch,x_patch
###########################################################################################################################################
        '''
        a=1
  
        #############################x_mean#################################
        x_red=x_mean[0,0,:,:]+a*backward(masked_data[0,0,:,:]-forward(x_mean[0,0,:,:],mask),mask)      #x_mean---x_mean_patch;   mask_data---cut_F_masked_data;  mask_data_mean---cut_F_masked_data_men
        x_green=x_mean[0,1,:,:]+a*backward(masked_data[0,1,:,:]-forward(x_mean[0,1,:,:],mask),mask)
        x_blue=x_mean[0,2,:,:]+a*backward(masked_data[0,2,:,:]-forward(x_mean[0,2,:,:],mask),mask)
        x_mean=torch.stack((x_red,x_green,x_blue),dim=0).unsqueeze(0)

        ###############################x####################################
        x_red=x_mean[0,0,:,:]+backward(masked_data_mean[0,0,:,:]-forward(x_mean[0,0,:,:],mask),mask)
        x_green=x_mean[0,1,:,:]+backward(masked_data_mean[0,1,:,:]-forward(x_mean[0,1,:,:],mask),mask)
        x_blue=x_mean[0,2,:,:]+backward(masked_data_mean[0,2,:,:]-forward(x_mean[0,2,:,:],mask),mask)
        x=torch.stack((x_red,x_green,x_blue),dim=0).unsqueeze(0)
        x1,x2,x3=x,x,x
        '''
          
        ########################校正器更新#########################
        x1,x2,x3, x_mean_patch = corrector_update_fn(x1,x2,x3,x_mean_patch,vec_t,model=model)#校正器更新
         
################################################将tensor转化为hankel###############################################################
        x_mean_patch = x_mean_patch.clone().detach().squeeze(0)       #(322,192,192)
        A_new=torch.zeros((322*192,192),dtype=torch.float64)
        for i in range(322): 
          A_new[192*i:192*(i+1),:]=x_mean_patch[i,:,:]
        A_no=A[322*192:,:]
        P_output=torch.cat((A_new.cuda(),A_no),0) # (62001, 192)          #将训练好的hankel矩阵和没训练的hankel矩阵部分进行拼接
###################################################################################################################
##############################################hankel转化为图像#############################################
        P_output = reshape_fortran(P_output,[P_output.shape[0],int(P_output.shape[1]/3),3])   #（62001,64,3）   
        kcomplex_h = row2im(P_output, size_data,ksize )         #(256,256,3)

       # kcomplex_h = (kcomplex_h-kcomplex_h.min())/(kcomplex_h.max()-kcomplex_h.min())
##############################################################################################################
        ####################################x_mean保真################################################
          
        x_red=kcomplex_h[:,:,0]+a*backward(masked_data[:,:,0]-forward(kcomplex_h[:,:,0],mask),mask)    #注意backward（）中的cut_F_masked_data[j,:,:,0]是频域的
  
        #x_mean---x_mean_patch;   mask_data---cut_F_masked_data;  mask_data_mean---cut_F_masked_data_men;    mask---cut_mask;
 
        x_green=kcomplex_h[:,:,1]+a*backward(masked_data[:,:,1]-forward(kcomplex_h[:,:,1],mask),mask) 
        x_blue=kcomplex_h[:,:,2]+a*backward(masked_data[:,:,2]-forward(kcomplex_h[:,:,2],mask),mask)
        kcomplex_h=torch.stack((x_red,x_green,x_blue),dim=2)        #(64,64,3)
          
        ####################################x保真#####################################################
          
        x_red=kcomplex_h[:,:,0]+backward(masked_data_mean[:,:,0]-forward(kcomplex_h[:,:,0],mask),mask)     #注意backward（）中的cut_F_masked_data_mean[j,:,:,0]是频域的              
        x_green=kcomplex_h[:,:,1]+backward(masked_data_mean[:,:,1]-forward(kcomplex_h[:,:,1],mask),mask)
        x_blue=kcomplex_h[:,:,2]+backward(masked_data_mean[:,:,2]-forward(kcomplex_h[:,:,2],mask),mask)
        x=torch.stack((x_red,x_green,x_blue),dim=2)                  #(64,64,3)
          
#############################################图像转化为hankel############################################################
       # kcomplex_h = (kcomplex_h-kcomplex_h.min())/(kcomplex_h.max()-kcomplex_h.min())
        hankel = im2row(kcomplex_h,ksize)           #(62001,64,3)       转化成hankel矩阵
        size_temp = hankel.shape
        A = reshape_fortran(hankel,[size_temp[0],size_temp[1]*size_temp[2]])    #(62001,192)

##############################################hankel转化为tensor##################################################################
        ans_1 = torch.zeros((322,192,192),dtype=torch.float64)
        for i in range(322):
          cut = A[i*192:(i+1)*192,:]
          ans_1[i,:,:] = cut          #(322,192,192)
        x_mean_patch=ans_1.to(torch.float32).cuda().unsqueeze(0)
        '''
        x_0=tvdenoise(x[:,:,0],10,2)#TV去噪
        x_1=tvdenoise(x[:,:,1],10,2)
        x_2=tvdenoise(x[:,:,2],10,2)
        x=torch.stack((x_0,x_1,x_2),dim=2)
        '''
        save_x = x.cpu().numpy()                       #######################
        print(save_x.min(),'---',save_x.max())
        save_x = (save_x-save_x.min())/(save_x.max()-save_x.min())
       # save_x[57:200,57:200,:] = save_x[57:200,57:200,:]-save_x.min()
        data_ori = data[0,:,:,:].cpu().numpy().transpose(1,2,0)     #numpy:(256,256,3)
        data_ori = (data_ori-data_ori.min())/(data_ori.max()-data_ori.min())
      #  save_x = np.clip(save_x,0,1)
      #  data_ori=np.clip(data_ori,0,1)
        '''
        data_ori_patch = data_ori[57:200,57:200,:]
        save_x_patch = save_x[57:200,57:200,:] 

        psnr=compare_psnr(np.abs(save_x_patch)*255,np.abs(data_ori_patch)*255,data_range=255)
        ssim=compare_ssim(np.abs(save_x_patch),np.abs(data_ori_patch),multichannel=True,data_range=1)
        '''
        psnr=compare_psnr(np.abs(save_x)*255,np.abs(data_ori)*255,data_range=255)
        ssim=compare_ssim(np.abs(save_x),np.abs(data_ori),multichannel=True,data_range=1)
        
        save_x = save_x.astype(np.float32)
        save_x = cv2.cvtColor(save_x,cv2.COLOR_BGR2RGB)
        cv2.imwrite('patch_{}.png'.format(0),save_x*255)
        print('内循环:',n,'psnr:',psnr,'ssim:',ssim)

        '''
        #############################x_mean#################################
        x_red=x_mean[0,0,:,:]+a*backward(masked_data[0,0,:,:]-forward(x_mean[0,0,:,:],mask),mask)
        x_green=x_mean[0,1,:,:]+a*backward(masked_data[0,1,:,:]-forward(x_mean[0,1,:,:],mask),mask)
        x_blue=x_mean[0,2,:,:]+a*backward(masked_data[0,2,:,:]-forward(x_mean[0,2,:,:],mask),mask)
        x_mean=torch.stack((x_red,x_green,x_blue),dim=0).unsqueeze(0)

        #############################x#################################
        x_red=x_mean[0,0,:,:]+backward(masked_data_mean[0,0,:,:]-forward(x_mean[0,0,:,:],mask),mask)
        x_green=x_mean[0,1,:,:]+backward(masked_data_mean[0,1,:,:]-forward(x_mean[0,1,:,:],mask),mask)
        x_blue=x_mean[0,2,:,:]+backward(masked_data_mean[0,2,:,:]-forward(x_mean[0,2,:,:],mask),mask)
        x=torch.stack((x_red,x_green,x_blue),dim=0).unsqueeze(0)
        '''
        
        
        # x1,x2,x3=x,x,x

####################################################################################################################
      #x_mean_save=np.clip(x_mean_save,0,1)
      #x_mean_save = (x_mean_save-x_mean_save.min())/(x_mean_save.max()-x_mean_save.min())
      #cv2.imwrite('fza_inpainting_3noise_0.2_FOV_z=12.5.png',x_mean_save*255)
    return save_x                           #    ,psnr_max,ssim_max

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
