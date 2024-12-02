Paper: One-sample Diffusion Modeling in Projection Domain for Low-dose CT Imaging (OSDM)

**Authors**: Bin Huang, Shiyu Lu, Liu Zhang, Boyu Lin, Weiwen Wu, Member, IEEE Qiegen Liu, Senior Member, IEEE

The code and the algorithm are for non-comercial use only.
Copyright 2023, School of Information Engineering, Nanchang University.

Low-dose computed tomography (CT) is crucial in clinical applications for reducing radiation risks. However, lowering the radiation dose will significantly degrade the image quality. In the meanwhile, common deep learning methods require large data, which are short for privacy leaking, expensive, and time-consuming. Therefore, we propose a fully unsupervised one-sample diffusion modeling (OSDM) in projection domain for low-dose CT reconstruction. To extract sufficient prior information from a single sample, the Hankel matrix formulation is employed. Besides, the penalized weighted least-squares and total variation are introduced to achieve superior image quality. Firstly, we train a score-based diffusion model on one sinogram to capture the prior distribution with input tensors extracted from the structural-Hankel matrix. Then, at inference, we perform iterative stochastic differential equation solver and data-consistency steps to obtain sinogram data, followed by the filtered back-projection algorithm for image reconstruction. The results approach normal-dose counterparts, validating OSDM as an effective and practical model to reduce artifacts while preserving image quality.


## The OSDM training process
![图片e](https://github.com/yqx7150/OSDM/assets/26964726/18d09431-9165-4b65-a76a-7613ff0e69c2)

   
## The pipeline of iterative reconstruction procedure in OSDM
![图片f](https://github.com/yqx7150/OSDM/assets/26964726/2255553f-b9ef-4a16-9a1b-b333d86899b7)


## Reconstruction results from 1e5 noise level using different methods.
![图片g](https://github.com/yqx7150/OSDM/assets/26964726/ae7a4585-0a3e-4646-8df1-406934f0f3d1)
## (a) GT&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (b) FBP, (c) SART-TV, (d) CNN, (e) U-Net, (f) NCSN++, and (g) OSDM.

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


## Other Related Projects  
<div align="center"><img src="https://github.com/yqx7150/OSDM/blob/main/All-CT.png" >  </div>   
    
  * REDAEP: Robust and Enhanced Denoising Autoencoding Prior for Sparse-View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9076295)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/REDAEP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9703672)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EASEL)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)
   
  * Generative Modeling in Sinogram Domain for Sparse-view CT Reconstruction      
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/10233041)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/GMSD)

  * DP-MDM: Detail-Preserving MR Reconstruction via Multiple Diffusion Models  
[<font size=5>**[Paper]**</font>](http://arxiv.org/abs/2211.13857)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DP-MDM)
     
  * MSDiff: Multi-Scale Diffusion Model for Ultra-Sparse View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://arxiv.org/pdf/2405.05763)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MSDiff)
    
  * Wavelet-improved score-based generative model for medical imaging  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10288274)       
       
  * 基于深度能量模型的低剂量CT重建  
[<font size=5>**[Paper]**</font>](http://cttacn.org.cn/cn/article/doi/10.15953/j.ctta.2021.077)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EBM-LDCT)  
