import os
import numpy as np
import cv2 as cv
import time
# import tensorflow as tf

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import test_split1
import english
import chinese
#import english_only_model
import locate_hsv_plate
import hsv
# import recognition
import video


picture = 0
clear = 1

#path = "video/plate4.mp4"#"images2/plate0.jpg"#"picture_data/1/1/01-02.jpg"
# path = "images/plate1.jpg"
def predict_total(rvtime,predict_result,predict_result0,path,picture,clear=0,choose=0):
    file_name = "result.jpg" # '../' + list1[i]
    pic_name = "mypic"
    time1 = 0
    time2 = 0
    #text = ""
    if os.path.exists(file_name):
        os.remove(file_name)
    for i in range(7):
        if os.path.exists(pic_name + str(i) + ".jpg"):
            os.remove(pic_name + str(i) + ".jpg")

    if(picture):
        if(os.path.exists(path)):
            hsv.get_candidate_paltes_by_hsv(path)
            locate_hsv_plate.locate(path)
            #recognition.locate("picture_data/1/1/01-01.jpg")
            if (os.path.exists("result.jpg")):
                split0, split_num = test_split1.split_licensePlate_character(test_split1.remove_plate_upanddown_border("result.jpg", clear))
                #split_num = np.array(split0).shape
                #split_num = split_num[0]
                # print(split_num)

                start1 = time.perf_counter()
                #神经网络
                chinese.model1(split_num,predict_result)
                #english_only_model.model3(split_num)
                english.model2(split_num,predict_result)
                end1 = time.perf_counter()
                time1 = end1-start1


            else:
                print("未识别出车牌")
    else:
        picture_num = video.getvideo(path, clear)



if __name__ == "__main__":
    picture = 1
    clear = 1
    predict_result = []
    rvtime = []
    predict_result0 = []
    # path = "video/plate_video.mp4"
    # #"images2/plate0.jpg"#"picture_data/1/1/01-02.jpg"
    path = "images2/plate4.jpg"
    # for item in os.listdir(path):
    #     pathdir = os.path.join(path, item)
    #     print(pathdir)
   # (rvtime, predict_result, predict_result0, path, picture, clear=0, choose=0)
    predict_total(rvtime, predict_result, predict_result0, path, picture, clear)