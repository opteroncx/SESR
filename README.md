# SESR
SESR: Single Image Super Resolution with Recursive Squeeze and Excitation Networks (Submitted to ICPR 2018)
https://arxiv.org/abs/1801.10319
---
![](https://github.com/opteroncx/SESR/raw/master/figures/f1.png)  
### Quality for scale x4 and x8
    Trained on yang91+bsd200, the default recursion depth for each branch is set to 4
---
| DataSet/Method        | PSNR/SSIM x4| PSNR/SSIM x8|
| ------------- | -----| -----:|
| Set5      | 31.84/ 0.891      |26.35/ 0.742      |
| Set14     | 28.32/ 0.784      |24.41/ 0.637      | 
| BSD100    | 27.42/ 0.737      |24.65/ 0.602      | 
| Urban100  | 25.42/ 0.771      |  21.82/ 0.595      | 
---
    Trained on div2k, r=4
---
| DataSet/Method        | PSNR/SSIM|
| ------------- | -----:|
| Set5      | 32.11/ 0.895      |
| Set14     | 28.47/ 0.787      | 
| BSD100    | 27.53/ 0.739      | 
| Urban100    | 25.86/ 0.784      | 
### Compare with other methods
![](https://github.com/opteroncx/SESR/raw/master/figures/f2.png)  
### Requirement
    Python 2.7
    Pytorch 0.2.0
    opencv-python
    numpy
### Train
    python train.py --cuda
### Evaluate
    python test.py --cuda
### Do Super resolution on your own images
    python test.py --cuda --mode sr --testdir path_to_your_image

Reference
---
https://github.com/twtygqyy/pytorch-LapSRN
https://github.com/jiny2001/dcscn-super-resolution
https://github.com/jmiller656/EDSR-Tensorflow
https://github.com/grevutiu-gabriel/python-ssim/blob/master/python-ssim.py
