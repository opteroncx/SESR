# SESR
SESR: Single Image Super Resolution with Recursive Squeeze and Excitation Networks
---
![](https://github.com/opteroncx/SESR/raw/master/f1s.png)  
### Quality for scale x4
| DataSet/Method        | PSNR/SSIM|
| ------------- | -----:|
| Set5      | 31.84/ 0.891      |
| Set14     | 28.32/ 0.784      | 
| BSD100    | 27.42/ 0.737      | 
| Urban100    | 25.42/ 0.771      | 
### Compare with other methods
![](https://github.com/opteroncx/SESR/raw/master/fig.png)  
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
