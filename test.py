# -*- coding:utf-8 -*-
import argparse
import torch
import os
import cv2
import pyssim
import codecs
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import as_strided as ast
from PIL import Image
from torch.autograd import Variable
import numpy as np
import time, math
import scipy.io as sio
from skimage import measure

parser = argparse.ArgumentParser(description="Test or Super resolution with SESR")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_100.pth", type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--testdir", default='all', type=str, help="")
parser.add_argument("--mode", default="evaluate", type=str, help="")

opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

def savelog(path,psnr,ssim):
    log_path='./log/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    test_time=time.time()
    test_time=str(int(test_time))
    log=codecs.open(log_path+'test_log'+'.txt','a+','utf-8')
    log.writelines("=======================================\n")
    log.writelines(test_time+'\n')
    log.writelines(path+'\n')
    log.writelines('PSNR==>%f  \n'%psnr)
    log.writelines('SSIM==>%f  \n'%ssim)
    log.close()


def eval():
    if opt.testdir == 'all':
        # run all tests
        testdirs=["testdir/Set5","testdir/Set14","testdir/Bsd100","testdir/Urban100"]
        for t in testdirs:
            evaluate_by_path(t)
    else:
        t=opt.testdir
        evaluate_by_path(t)


def evaluate_by_path(path):
    pimages=os.listdir(path)
    s_psnr=0
    s_ssim=0
    save=True
    eva=True
    convert=True
    for pimg in pimages:
        img = np.array(Image.open(path+'/'+pimg))
        psnr,ssim=predict(img,save,convert,eva,pimg)
        s_psnr+=psnr
        s_ssim+=ssim
    avg_psnr=s_psnr/len(pimages)
    avg_ssim=s_ssim/len(pimages)
    print_summary(avg_psnr,avg_ssim)
    savelog(path,avg_psnr,avg_ssim)

def sr():
    path=opt.testdirs
    pimages=os.listdir(path)
    save=True
    eva=False
    convert=True
    for pimg in pimages:
        img=cv2.imread(path+'/'+pimg)
        predict(img,save,convert,eva,pimg)

def predict(img_read,save,convert,eva,name):
    if convert:
        if eva:
            h,w,_=img_read.shape
            im_gt_y=convert_rgb_to_y(img_read)
            gt_yuv=convert_rgb_to_ycbcr(img_read)
            im_gt_y=im_gt_y.astype("float32")
            sc=1.0/opt.scale
            img_y=resize_image_by_pil(im_gt_y,sc)
            img_y=img_y[:,:,0]
            im_gt_y=im_gt_y[:,:,0]
        else:
            img_y=convert_rgb_to_y(img_read)
    else:
        im_gt_y,img_y=img_read
        im_gt_y=im_gt_y.astype("float32")
    im_input = img_y/255.
    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
    model = torch.load(opt.model)["model"]
    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    HR_2x, HR_4x = model(im_input)
    elapsed_time = time.time() - start_time
    if opt.scale ==2:
        HR_2x = HR_2x[-1].cpu()
        im_h_y = HR_2x.data[0].numpy().astype(np.float32)
    elif opt.scale ==4:
        HR_4x = HR_4x[-1].cpu()
        im_h_y = HR_4x.data[0].numpy().astype(np.float32)
    else:
        print('input wrong scale')

    im_h_y = im_h_y*255.
    im_h_y[im_h_y<0] = 0
    im_h_y[im_h_y>255.] = 255.
    im_h_y = im_h_y[0,:,:]
    if save:
        recon=convert_y_and_cbcr_to_rgb(im_h_y, gt_yuv[:, :, 1:3])
        save_figure(recon,name)
    if eva:
        #PSNR and SSIM
        psnr_predicted = PSNR(np.uint8(im_gt_y), np.uint8(im_h_y),shave_border=opt.scale)
        ssim_predicted = pyssim.compute_ssim(im_gt_y, im_h_y)
        print("test psnr/ssim=%f/%f"%(psnr_predicted,ssim_predicted))
        return psnr_predicted,ssim_predicted
    else:
        print("doing super resolution")

def print_summary(psnr,ssim):
    print("Scale=",opt.scale)
    print("PSNR=", psnr)
    print("SSIM=",ssim)


def save_figure(img,name):
    out_path='./out/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    print 'saved '+name
    img = np.uint8(img)
    im = Image.fromarray(img)
    im.save(out_path+name[:-4]+'.png')

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    return measure.compare_psnr(gt,pred,255)

def convert_rgb_to_y(image, jpeg_mode=False, max_value=255.0):
    if len(image.shape) <= 2 or image.shape[2] == 1:
        return image

    if jpeg_mode:
        xform = np.array([[0.299, 0.587, 0.114]])
        y_image = image.dot(xform.T)
    else:
        xform = np.array([[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0]])
        y_image = image.dot(xform.T) + (16.0 * max_value / 256.0)

    return y_image


def convert_rgb_to_ycbcr(image, jpeg_mode=False, max_value=255):
    if len(image.shape) < 2 or image.shape[2] == 1:
        return image

    if jpeg_mode:
        xform = np.array([[0.299, 0.587, 0.114], [-0.169, - 0.331, 0.500], [0.500, - 0.419, - 0.081]])
        ycbcr_image = image.dot(xform.T)
        ycbcr_image[:, :, [1, 2]] += max_value / 2
    else:
        xform = np.array(
            [[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0], [- 37.945 / 256.0, - 74.494 / 256.0, 112.439 / 256.0],
             [112.439 / 256.0, - 94.154 / 256.0, - 18.285 / 256.0]])
        ycbcr_image = image.dot(xform.T)
        ycbcr_image[:, :, 0] += (16.0 * max_value / 256.0)
        ycbcr_image[:, :, [1, 2]] += (128.0 * max_value / 256.0)

    return ycbcr_image


def convert_y_and_cbcr_to_rgb(y_image, cbcr_image, jpeg_mode=False, max_value=255.0):

    if len(y_image.shape) == 3 and y_image.shape[2] == 3:
        y_image = y_image[:, :, 0:1]

    ycbcr_image = np.zeros([y_image.shape[0], y_image.shape[1], 3])
    ycbcr_image[:, :, 0] = y_image
    ycbcr_image[:, :, 1:3] = cbcr_image[:, :, 0:2]

    return convert_ycbcr_to_rgb(ycbcr_image, jpeg_mode=jpeg_mode, max_value=max_value)


def convert_ycbcr_to_rgb(ycbcr_image, jpeg_mode=False, max_value=255.0):
    rgb_image = np.zeros([ycbcr_image.shape[0], ycbcr_image.shape[1], 3])  # type: np.ndarray

    if jpeg_mode:
        rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - (128.0 * max_value / 256.0)
        xform = np.array([[1, 0, 1.402], [1, - 0.344, - 0.714], [1, 1.772, 0]])
        rgb_image = rgb_image.dot(xform.T)
    else:
        rgb_image[:, :, 0] = ycbcr_image[:, :, 0] - (16.0 * max_value / 256.0)
        rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - (128.0 * max_value / 256.0)
        xform = np.array(
            [[max_value / 219.0, 0, max_value * 0.701 / 112.0],
             [max_value / 219, - max_value * 0.886 * 0.114 / (112 * 0.587), - max_value * 0.701 * 0.299 / (112 * 0.587)],
             [max_value / 219.0, max_value * 0.886 / 112.0, 0]])
        rgb_image = rgb_image.dot(xform.T)

    return rgb_image

def resize_image_by_pil(image, scale, resampling_method="bicubic"):
    width, height = image.shape[1], image.shape[0]
    new_width = int(width * scale)
    new_height = int(height * scale)

    if resampling_method == "bicubic":
        method = Image.BICUBIC
    elif resampling_method == "bilinear":
        method = Image.BILINEAR
    elif resampling_method == "nearest":
        method = Image.NEAREST
    else:
        method = Image.LANCZOS

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # the image may has an alpha channel
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    else:
        image = Image.fromarray(image.reshape(height, width))
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
        image = image.reshape(new_height, new_width, 1)
    return image

##################################
def main():
    if opt.mode=="evaluate":
        eval()
    else:
        sr()

main()
