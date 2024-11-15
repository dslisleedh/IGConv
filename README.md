# IGConv: Implicit Grid Convolution for Multi-Scale Image Super-Resolution

![image](https://github.com/dslisleedh/IGConv/blob/main/figs/OverallArchitecture.png)

>For Image Super-Resolution (SR), it is common to train and evaluate scale-specific models composed of an encoder and upsampler for each targeted scale. 
Consequently, many SR studies encounter substantial training times and complex deployment requirements.
In this paper, we address this limitation by training and evaluating multiple scales simultaneously. 
Notably, we observe that encoder features are similar across scales and that the Sub-Pixel Convolution (SPConv), widely-used scale-specific upsampler, exhibits strong inter-scale correlations in its functionality.
Building on these insights, we propose a multi-scale framework that employs a single encoder in conjunction with Implicit Grid Convolution (IGConv), our novel upsampler, which unifies SPConv across all scales within a single module.
Extensive experiments demonstrate that our framework achieves comparable performance to existing fixed-scale methods while reducing the training budget and stored parameters three-fold and maintaining the same latency. 
Additionally, we propose IGConv$^{+}$ to improve performance further by addressing spectral bias and allowing input-dependent upsampling and ensembled prediction. 
As a result, ATD-IGConv$^{+}$ achieves a notable 0.21dB improvement in PSNR on Urban100$\times$4, while also reducing the training budget, stored parameters, and inference cost compared to the existing ATD.

This repository is an official implementation of the paper "Implicit Grid Convolution for Multi-Scale Image Super-Resolution", Arxiv, 2024.

by Dongheon Lee, Seokju Yun, and Youngmin Ro

[[Paper]](https://arxiv.org/abs/2408.09674), [[PretrainedModels]](https://github.com/dslisleedh/IGConv/releases/tag/v1.0.0)

## Installation

```bash
conda create -n igconv python=3.10
conda activate igconv
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt 
python setup.py develop
```

### Optional packages (for MambaIR)
```bash
pip install causal_conv1d
pip install mamba_ssm
```
Unlike pytorch which utilizes downloaded(whl or conda) cuda-runtime, these packages utilize "local" cuda-runtime.
So, install these carefully.

### NOTE
Do Not Install BasicSR using PIP !!! 

## Train

### Single GPU (NOT RECOMMENDED!!!)
```bash
python igconv/train.py -opt $CONFIG_PATH
```
Gradient accumulation for averaging gradients over multiple sub-batches is not implemented in this code. 
Therefore, we highly recommend running our code on multiple GPUs.
In all cases, we used 4 GPUs (RTX3090 or A6000).

### Multi-GPU (Local)
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch\
  --nproc_per_node=4 --master_port=25000 \
 igconv/train.py -opt $CONFIG_PATH --launcher pytorch
```

### Multi-GPU (SLURM)
```bash
PYTHONPATH="./:${PYTHONPATH}" \
GLOG_vmodule=MemcachedClient=-1 \
srun -p $PARTITION --mpi=pmi2 --gres $GPU --ntasks=4 --cpus-per-task 16 --kill-on-bad-exit=1 \
python -u igconv/train.py -opt $CONFIG_PATH --launcher="slurm"
```

## Test

```bash
python igconv/test.py -opt $CONFIG_PATH
```

## Results

![image](https://github.com/dslisleedh/IGConv/blob/main/figs/teaser.png)

<details>
<summary>Tables</summary>

![image](https://github.com/dslisleedh/IGConv/blob/main/figs/Quantitative_DIV2K.png)
![image](https://github.com/dslisleedh/IGConv/blob/main/figs/Quantitative_DF2K.png)
![image](https://github.com/dslisleedh/IGConv/blob/main/figs/Quantitative_PT.png)
</details>

## License
This project is released under the MIT license.

## Acknowledgement
Our work is based on the implementation of many studies, and we are very grateful to the authors of [these studies](https://github.com/dslisleedh/IGConv/blob/main/licences/readme.md). 
- BasicSR
- NeoSR
- DySample
- SwinIR
- SRFormer
- HiT-SR
- EDSR
- RCAN
- RDN
- SMFANet
- MambaIR
- HAT
- ATD
