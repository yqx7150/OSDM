Paper: One-sample Diffusion Modeling in Projection Domain for Low-dose CT Imaging

**Authors**: Bin Huang, Shiyu Lu, Liu Zhang, Boyu Lin, Weiwen Wu, Member, IEEE Qiegen Liu, Senior Member, IEEE
IEEE Transactions on Radiation and Plasma Medical Sciences, vol. 8, no. 8, pp. 902-915, Nov. 2024.    
https://ieeexplore.ieee.org/document/10506793                  
     
The code and the algorithm are for non-comercial use only.
Copyright 2023, School of Information Engineering, Nanchang University.

Low-dose computed tomography (CT) is crucial in clinical applications for reducing radiation risks. However, lowering the radiation dose will significantly degrade the image quality. In the meanwhile, common deep learning methods require large data, which are short for privacy leaking, expensive, and time-consuming. Therefore, we propose a fully unsupervised one-sample diffusion modeling (OSDM) in projection domain for low-dose CT reconstruction. To extract sufficient prior information from a single sample, the Hankel matrix formulation is employed. Besides, the penalized weighted least-squares and total variation are introduced to achieve superior image quality. Firstly, we train a score-based diffusion model on one sinogram to capture the prior distribution with input tensors extracted from the structural-Hankel matrix. Then, at inference, we perform iterative stochastic differential equation solver and data-consistency steps to obtain sinogram data, followed by the filtered back-projection algorithm for image reconstruction. The results approach normal-dose counterparts, validating OSDM as an effective and practical model to reduce artifacts while preserving image quality.


## The OSDM training process
![图片e](https://github.com/yqx7150/OSDM/assets/26964726/18d09431-9165-4b65-a76a-7613ff0e69c2)

   
## The pipeline of iterative reconstruction procedure in OSDM
![图片f](https://github.com/yqx7150/OSDM/assets/26964726/2255553f-b9ef-4a16-9a1b-b333d86899b7)


## Reconstruction results from 1e5 noise level using different methods.
![图片g](https://github.com/yqx7150/OSDM/assets/26964726/ae7a4585-0a3e-4646-8df1-406934f0f3d1)
## &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(a) GT &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (b) FBP &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(c) SART-TV &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(d) CNN &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(e) U-Net &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(f) NCSN++ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (g) OSDM

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


### Other Related Projects
<div align="center"><img src="https://github.com/yqx7150/OSDM/blob/main/All-CT.png" >  </div>   
    
  * Generative Modeling in Sinogram Domain for Sparse-view CT Reconstruction      
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/10233041)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/GMSD)

  * One Sample Diffusion Model in Projection Domain for Low-Dose CT Imaging  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10506793)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/OSDM)

  * Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9703672)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EASEL)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)
    
  * REDAEP: Robust and Enhanced Denoising Autoencoding Prior for Sparse-View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9076295)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/REDAEP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)

  * Wavelet-improved score-based generative model for medical imaging  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10288274)

  * 基于深度能量模型的低剂量CT重建  
[<font size=5>**[Paper]**</font>](http://cttacn.org.cn/cn/article/doi/10.15953/j.ctta.2021.077)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EBM-LDCT)  

 * Stage-by-stage Wavelet Optimization Refinement Diffusion Model for Sparse-view CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10403850)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/SWORD)

  * Dual-Domain Collaborative Diffusion Sampling for Multi-Source Stationary Computed Tomography Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/10577271)   [<font size=5>**[Code]**</font>](https://github.com/lizrzr/DCDS-Dual-domain_Collaborative_Diffusion_Sampling)

  * Low-rank Angular Prior Guided Multi-diffusion Model for Few-shot Low-dose CT Reconstruction     
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10776993)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PHD)

  * Physics-informed DeepCT: Sinogram Wavelet Decomposition Meets Masked Diffusion  
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2501.09935)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/SWARM)    
                    
  * MSDiff: Multi-Scale Diffusion Model for Ultra-Sparse View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://arxiv.org/pdf/2405.05763)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MSDiff)

  * Ordered-subsets Multi-diffusion Model for Sparse-view CT Reconstruction      
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2505.09985)
                          
  * Virtual-mask Informed Prior for Sparse-view Dual-Energy CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2504.07753)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VIP-DECT)

  * Raw_data_generation  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Raw_data_generation)

  * PRO: Projection Domain Synthesis for CT Imaging  [<font size=5>**[Paper]**</font>](https://arxiv.org/pdf/2506.13443)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PRO)
       
  * UniSino: Physics-Driven Foundational Model for Universal CT Sinogram Standardization[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2508.17816)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/UniSino)

  * Diffusion Models for Medical Imaging
[<font size=5>**[Paper]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HKGM/tree/main/PPT) 


