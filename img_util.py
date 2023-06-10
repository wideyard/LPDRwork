'''
对粗定位的车牌进行透视矫正（已实现）并进一步定位（未实现）
'''


import os
import cv2
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import imutils
from skew_detection import *
from scipy import signal

matplotlib.use('TkAgg')


# 使用matplotlib显示图像
def show_img(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


# 灰度拉伸
def gray_stretch(gray, mode='linear'):
    if mode == 'linear':
        (max_value, min_value) = (np.max(gray), np.min(gray))
        gray = (255 * ((gray - min_value) / (max_value - min_value)))
        gray = gray.astype("uint8")
        return gray


# 锐化
def sharpen(img, mode='conv'):
    if mode == 'conv':
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]], dtype=np.float32)
        sharpened = cv2.filter2D(img, cv2.CV_32F, kernel)
        sharpened = cv2.convertScaleAbs(sharpened)
        return sharpened
    if mode == 'USM':  # unsharpen mask
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        return sharpened


def verifyPlateByVerticalProjection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    gray = gray_stretch(gray)
    # thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_BINARY_INV)
    show_img(thresh)
    h, w = thresh.shape[:2]

    projection_v =  h - thresh.sum(axis=0) / 256
    projection_v_inv = projection_v * -1
    projection_v_smooth = signal.savgol_filter(projection_v, 11, 3)
    projection_v_inv_smooth = signal.savgol_filter(projection_v_inv, 11, 3)
    plt.plot(projection_v_smooth)
    plt.plot(projection_v_inv_smooth)
    plt.show()

    projection_h = w - thresh.sum(axis=1) / 256
    projection_h_inv = projection_h * -1
    projection_h_smooth = signal.savgol_filter(projection_h, 11, 3)
    projection_h_inv_smooth = signal.savgol_filter(projection_h_inv, 11, 3)
    plt.plot(projection_h_smooth)
    plt.plot(projection_h_inv_smooth)
    plt.show()

    peaks_v_idx, peaks_v_height = signal.find_peaks(projection_v_smooth, height=40, distance = 45)
    peaks_v_idx_inv, peaks_v_height_inv = signal.find_peaks(projection_v_inv_smooth, height=-20, distance = 45)
    if len(peaks_v_idx) < 7 - 2 or len(peaks_v_idx) > 7 + 2:
        return False
    if len(peaks_v_idx_inv) < 7 - 2 or len(peaks_v_idx_inv) > 7 + 2:
        return False
    return True


img_path = os.path.join('origin_images', 'plate_saved3.jpg')
img = cv2.imread(img_path)
# img_path = os.path.join('G:/code/pro/num_plate_recognision/Tensorflow/workspace/images/train','02-92_84-253&504_503&596-500&594_257&581_255&507_498&520-0_0_4_0_25_32_24-86-30.jpg')
# img = cv2.imread(img_path)
show_img(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = gray_stretch(gray)
skew_h, skew_v = skew_detection(gray)
print(skew_h, skew_v)
corr_img = v_rot(img, (90 - skew_v + skew_h), img.shape, 60);
corr_img = h_rot(corr_img, skew_h)
show_img(corr_img)
verifyPlateByVerticalProjection(corr_img)

