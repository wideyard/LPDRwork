import os
import pandas as pd
import cv2
import numpy as np
from pathlib import Path


TRAIN_DIR = 'data/CCPD2019/ccpd_base'

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

img_name = 'data/CCPD2019/ccpd_base/01-86_91-298&341_449&414-458&394_308&410_304&357_454&341-0_0_14_28_24_26_29-124-24'
iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
print(iname)
print(iname[4])
car_name = iname[4].split("_")
print(car_name)
labels = []
for i in range(len(car_name)):
    if i==0:
        print(provincelist[int(car_name[0])])
        labels.append(provincelist[int(car_name[0])])
    else:
        print(wordlist[int(car_name[i])])
        labels.append(wordlist[int(car_name[i])])


a = np.zeros((3, 4))
# str = os.'dataset/chinese/' + str(labels[i]) + os.sep + str(i) + '.jpg'

print(labels[0])

# str = os.path.join('dataset/chinese/', Path(str(labels[0]))) + os.sep + str(0) + '.jpg'
str1 = os.path.join('dataset/chinese/', Path(str(labels[0])))
print(str1)
if not os.path.exists(str1):
    os.makedirs(str1)
path = str1 + os.sep + str(len(os.listdir(str1))) + '.jpg'
# cv2.imwrite('dataset/chinese/{}/{}.jpg'.format(str(labels[i]),str(i)), a)
cv2.imwrite(path, a)


# import cv2
# import os
#
# img = cv2.imread('img.jpg', 1) # 读取img图片
#
# dirs = 'result/10'  # 想要保存的路径

# if not os.path.exists(dirs):   # 如果不存在路径，则创建这个路径，关键函数就在这两行，其他可以改变
#     os.makedirs(dirs)
#
# path = dirs+'/'+'img.jpg'  # 将文件路径与文件名进行结合，得到完整的文件路径以及名称
#
# cv2.imwrite(path, img)  # 保存图片到这个路径
