# InDuDoNet+: A Deep Unfolding Dual Domain Network for Metal Artifact Reduction in CT Images
[Hong Wang](https://hongwang01.github.io/), Yuexiang Li, Haimiao Zhang, [Deyu Meng](http://gr.xjtu.edu.cn/web/dymeng), [Yefeng Zheng](https://sites.google.com/site/yefengzheng/)

[[Google Drive]](https://drive.google.com/file/d/12NnSNk2aj-NE_MxpT-tJw2mj33cBcb4k/view?usp=sharing)[[Arxiv]](https://arxiv.org/pdf/2112.12660.pdf)

The conference paper is [InDuDoNet(MICCAI2021)](https://github.com/hongwang01/InDuDoNet)

## Abstract
During the computed tomography (CT) imaging process, metallic implants within patients always cause harmful artifacts, which adversely degrade the visual quality of reconstructed CT images and negatively affect the subsequent clinical diagnosis. For the metal artifact reduction (MAR) task, current deep learning based methods have achieved promising performance. However, most of them share two main common limitations: 1) the CT physical imaging geometry constraint is not comprehensively incorporated into deep network structures; 2) the entire framework has weak interpretability for the specific MAR task; hence, the role of every network module is difficult to be evaluated. To alleviate these issues, in the paper, we construct a novel interpretable dual domain network, termed InDuDoNet+,  into which CT imaging process is finely embedded. Concretely, we derive a joint spatial and Radon domain reconstruction model and propose an optimization algorithm with only simple operators for solving it. By unfolding the iterative steps involved in the proposed algorithm into the corresponding network modules, we easily build the InDuDoNet+ with clear interpretability. Furthermore, we analyze the CT values among different tissues, and merge the prior observations into a prior network for our InDuDoNet+, which significantly improve its generalization performance. Comprehensive experiments on synthesized data and clinical data substantiate the superiority of the proposed methods as well as the superior generalization performance beyond the current state-of-the-art (SOTA) MAR methods.

## Motivation
<div  align="center"><img src="figs/motivation.jpg" height="100%" width="100%" alt=""/></div>

## Knowledge-Driven Prior-net

<div  align="center"><img src="figs/priornet.jpg" height="100%" width="100%" alt=""/></div>


## Dependicies 
Refer to **environment.yml**. This repository is tested under the following system settings:

Python 3.10

Pytorch 2.0.1

CUDA 11.8

GPU NVIDIA A1000

**For running the code,  please first test whether ODL and Astra are both installed correctly. This is quite important.**

If using Anaconda, you may try the following steps:
```
conda create --name InDuDoNet python=3.10
conda activate InDuDoNet
conda install -c astra-toolbox/label/dev astra-toolbox
pip install https://github.com/yiluzhou1/odl/archive/refs/heads/master.zip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib h5py scikit-learn scikit-image nibabel pyyaml tqdm
```

Note: To avoid "libiomp5md.dll" error, you may need to search and rename "libiomp5md.dll" in your env folder


## Dataset & Training & Testing
Refer to [InDuDoNet](https://github.com/hongwang01/InDuDoNet) for the settings. 


## Training
```
CUDA_VISIBLE_DEVICES=0 python train.py --data_path "deeplesion/train/" --log_dir "logs" --model_dir "pretrained_model/"
```


## Testing
```
CUDA_VISIBLE_DEVICES=0 python test_deeplesion.py --data_path "deeplesion/test/" --model_dir "pretrained_model/InDuDoNet+_latest.pt" --save_path "results/deeplesion/" 
```

### For CLINIC-metal
```
CUDA_VISIBLE_DEVICES=0 python test_clinic.py --data_path "CLINIC_metal/test/" --model_dir "pretrained_model/InDuDoNet+_latest.pt" --save_path "results/CLINIC_metal/" --keep_originalshape False
```

## Experiments on Synthesized Data
<div  align="center"><img src="figs/deeplesion_boxplot.jpg" height="100%" width="100%" alt=""/></div>


## Experiments on SpineWeb
<div  align="center"><img src="figs/spine.jpg" height="100%" width="100%" alt=""/></div>



## Citations

```
@article{wang2023indudonet+,
  title={InDuDoNet+: A deep unfolding dual domain network for metal artifact reduction in CT images},
  author={Wang, Hong and Li, Yuexiang and Zhang, Haimiao and Meng, Deyu and Zheng, Yefeng},
  journal={Medical Image Analysis},
  volume={85},
  pages={102729},
  year={2023}
}
```

## Contact
If you have any question, please feel free to concat Hong Wang (Email: hongwang9209@hotmail.com)
