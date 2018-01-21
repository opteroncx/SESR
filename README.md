# SESR
PyTorch implementation of SESR: Single Image Super Resolution with Recursive Squeeze and Excitation Networks
---
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
