# IGConv: Implicit Grid Convolution for Multi-Scale Image Super-Resolution

![image](https://github.com/dslisleedh/IGConv/blob/main/figs/OverallArchitecture.png)

>Recently, Super-Resolution (SR) achieved significant performance improvement by employing neural networks. 
Most SR methods conventionally train a single model for each targeted scale, which increases redundancy in training and deployment in proportion to the number of scales targeted.
This paper challenges this conventional fixed-scale approach.
Our preliminary analysis reveals that, surprisingly, encoders trained at different scales extract similar features from images.
Furthermore, the commonly used scale-specific upsampler, Sub-Pixel Convolution (SPConv), exhibits significant inter-scale correlations.
Based on these observations, we propose a framework for training multiple integer scales simultaneously with a single model. 
We use a single encoder to extract features and introduce a novel upsampler, Implicit Grid Convolution (IGConv), which integrates SPConv at all scales within a single module to predict multiple scales.
Our extensive experiments demonstrate that training multiple scales with a single model reduces the training budget and stored parameters by one-third while achieving equivalent inference latency and comparable performance.
Furthermore, we propose IGConv+, which addresses spectral bias and input-independent upsampling and uses ensemble prediction to improve performance. 
As a result, SRFormer-IGConv+ achieves a remarkable 0.25dB improvement in PSNR at Urban100x4 while reducing the training budget, stored parameters, and inference cost compared to the existing SRFormer.

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

### DIV2K
![image](https://github.com/dslisleedh/IGConv/blob/main/figs/Quantitative_DIV2K.png)
### DF2K
![image](https://github.com/dslisleedh/IGConv/blob/main/figs/Quantitative_DF2K.png)
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
