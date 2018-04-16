# SESR
SESR: Single Image Super Resolution with Recursive Squeeze and Excitation Networks (To appear in ICPR 2018)
https://arxiv.org/abs/1801.10319
---
![](https://github.com/opteroncx/SESR/raw/master/figures/f1.png)  
### Quality for scale x4
---
    Trained on div2k, r=4
---
| DataSet/Method        | PSNR/SSIM|
| ------------- | -----:|
| Set5      | 32.05/ 0.897      |
| Set14     | 28.54/ 0.789     | 
| BSD100    | 27.51/ 0.743      | 
| Urban100    | 25.83/ 0.785      | 
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
---
