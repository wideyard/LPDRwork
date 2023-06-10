from plate_locate import plate_locaor
import numpy as np
from opencv_char_seperator import plate_char_seperator_save
import cv2
import os

IMAGE_DIR = 'data/CCPD2019/ccpd_base'

# provincelist = [
#     "皖", "沪", "津", "渝", "冀",
#     "晋", "蒙", "辽", "吉", "黑",
#     "苏", "浙", "京", "闽", "赣",
#     "鲁", "豫", "鄂", "湘", "粤",
#     "桂", "琼", "川", "贵", "云",
#     "西", "陕", "甘", "青", "宁",
#     "新"]
provincelist = [
    "wan", "hu", "jin", "yu1", "ji1",
    "jin", "meng", "liao", "ji2", "hei",
    "su", "zhe", "jing", "min", "gan1",
    "lu", "yu2", "e", "xiang", "yue",
    "gui1", "qiong", "chuan", "gui2", "yun",
    "xi", "shan", "gan2", "qing", "ning",
    "xin"]

wordlist = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9"]




image_dir = []
image_names = []
for item in os.listdir(IMAGE_DIR):
    image_dir.append(os.path.join(IMAGE_DIR, item))
    image_names.append(item)



def set_car_labels(image_name):
    iname = image_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
    labels = iname[4].split("_")
    car_labels=[]
    for i in range(len(labels)):
        if i == 0:
            car_labels.append(provincelist[int(labels[0])])
        else:
            car_labels.append(wordlist[int(labels[i])])

    return car_labels


for i in range(len(image_names)):
    hsv_candidate = plate_locaor.get_candidate_paltes_by_hsv(cv2.imread(image_dir[i]))
    sobel_candidate = plate_locaor.get_candidate_paltes_by_sobel(cv2.imread(image_dir[i]))

    for j in np.arange(len(hsv_candidate)):
        #img = hsv_candidate[j]
        img = hsv_candidate[j]
        #print(image_names[i])
        candidate_chars = plate_char_seperator_save.get_candidate_char(img, set_car_labels(image_names[i]))


