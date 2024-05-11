Paper: One-sample Diffusion Modeling in Projection Domain for Low-dose CT Imaging (OSDM)

**Authors**: Bin Huang, Shiyu Lu, Liu Zhang, Boyu Lin, Weiwen Wu, Member, IEEE Qiegen Liu, Senior Member, IEEE

The code and the algorithm are for non-comercial use only.
Copyright 2023, School of Information Engineering, Nanchang University.

Low-dose computed tomography (CT) is crucial in clinical applications for reducing radiation risks. However, lowering the radiation dose will significantly degrade the image quality. In the meanwhile, common deep learning methods require large data, which are short for privacy leaking, expensive, and time-consuming. Therefore, we propose a fully unsupervised one-sample diffusion modeling (OSDM) in projection domain for low-dose CT reconstruction. To extract sufficient prior information from a single sample, the Hankel matrix formulation is employed. Besides, the penalized weighted least-squares and total variation are introduced to achieve superior image quality. Firstly, we train a score-based diffusion model on one sinogram to capture the prior distribution with input tensors extracted from the structural-Hankel matrix. Then, at inference, we perform iterative stochastic differential equation solver and data-consistency steps to obtain sinogram data, followed by the filtered back-projection algorithm for image reconstruction. The results approach normal-dose counterparts, validating OSDM as an effective and practical model to reduce artifacts while preserving image quality.


## The OSDM training process
![](./configs/figg.png)  
   
## The pipeline of iterative reconstruction procedure in OSDM
![](configs/figg.png)  

## Reconstruction results from 1e5 noise level using different methods. (a) The reference image (b) FBP, (c) SART-TV, (d) CNN, (e) U-Net, (f) NCSN++, and (g) OSDM.
![](configs/figg.png)  

train:
python main.py --config=aapm_sin_ncsnpp_gb.py --workdir=exp --mode=train --eval_folder=result

test:
python PCsampling_demo.py

--workdir=exp_zl
--mode=train
--eval_folder=result
--config=aapm_sin_ncsnpp_gb.py

CUDA_VISIBLE_DEVICES=1 python main.py --config=aapm_sin_ncsnpp_gb.py --workdir=exp_zl --mode=train --eval_folder=result


vali[vali < 0] = 0

