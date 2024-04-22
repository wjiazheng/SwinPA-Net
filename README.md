# SwinPA-Net（T-NNLS 2024）
The code for the work [SwinPA-Net: Swin Transformer-Based Multiscale Feature Pyramid Aggregation Network for Medical Image Segmentation](https://ieeexplore.ieee.org/document/9895210)

## How to run
### 1. Environment
Please prepare an virtual environment with Python 3.6, and then use the command "pip install -r requirements.txt" for the dependencies.

### 2. dataset
Polyp datasets: we adopted the division method in [PraNet](https://github.com/DengPingFan/PraNet)  
[ISIC 2018 dataset](https://challenge.isic-archive.com/data/)

### 3. Train
python3 Train.py

### 3. Test
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
