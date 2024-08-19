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
