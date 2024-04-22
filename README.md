# SwinPA-Net（T-NNLS 2024）
The code for the work [SwinPA-Net: Swin Transformer-Based Multiscale Feature Pyramid Aggregation Network for Medical Image Segmentation](https://ieeexplore.ieee.org/document/9895210)

## How to run
### 1. Environment
Please prepare an virtual environment with Python 3.6, and then use the command "pip install -r requirements.txt" for the dependencies.

### 2. Dataset
Polyp datasets - we adopted the division method in [PraNet](https://github.com/DengPingFan/PraNet)  
[ISIC 2018 dataset](https://challenge.isic-archive.com/data/)

### 3. Pre-trained swin transformer model
The Pretrained models on ImageNet-1K ([Swin-T-IN1K](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth), [Swin-S-IN1K](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth), [Swin-B-IN1K](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth)) and ImageNet-22K ([Swin-B-IN22K](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth), [Swin-L-IN22K](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth)) are provided by [Swin Transformer](https://github.com/microsoft/Swin-Transformer?tab=readme-ov-file).

### 4. Train
python3 Train.py

### 5. Test
python3 Test.py

## Citation:
H. Du, J. Wang, M. Liu, Y. Wang and E. Meijering, "SwinPA-Net: Swin Transformer-Based Multiscale Feature Pyramid Aggregation Network for Medical Image Segmentation," in IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 4, pp. 5355-5366, April 2024, doi: 10.1109/TNNLS.2022.3204090.  

@ARTICLE{9895210,
  author={Du, Hao and Wang, Jiazheng and Liu, Min and Wang, Yaonan and Meijering, Erik},  
  journal={IEEE Transactions on Neural Networks and Learning Systems},   
  title={SwinPA-Net: Swin Transformer-Based Multiscale Feature Pyramid Aggregation Network for Medical Image Segmentation},   
  year={2024},  
  volume={35},  
  number={4},  
  pages={5355-5366},  
  keywords={Image segmentation;Transformers;Lesions;Task analysis;Monte Carlo methods;Semantics;Medical diagnostic imaging;Dense multiplicative connection (DMC) module;local pyramid attention (LPA) module;medical image segmentation},  
  doi={10.1109/TNNLS.2022.3204090}}
