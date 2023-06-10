"""
    对视频的相关操作
"""
import os

import cv2 as cv
import plate_Location
from split import Split_Char
import cnn_predict


# 从视频中截取帧
# 参数：传入一个视频路径
# 返回值，截取到的帧，图片集

def get_video_seperated(video_path):
    video = cv.VideoCapture(str(video_path))
    video_images = []
    times = []
    # 判断是否正常打开
    if video.isOpened():
        # 读帧
        success, frame = video.read()
    else:
        success = False
    i = 0
    # 设置固定帧率
    timeF = 35
    # 循环读取视频帧
    while success:
        # 读取成功
        if i % timeF == 0:
            # 将读到的图像放入图像集
            video_images.append(frame)
            milliseconds = video.get(cv.CAP_PROP_POS_MSEC)

            seconds = milliseconds // 1000
            milliseconds = milliseconds % 1000
            minutes = 0
            hours = 0
            if seconds >= 60:
                minutes = seconds // 60
                seconds = seconds % 60

            if minutes >= 60:
                hours = minutes // 60
                minutes = minutes % 60
            # print(int(hours), ":", int(minutes), ":", int(seconds), ":", int(milliseconds))
            time = "{}h,{}min,{}second,{}milliseconds".format(int(hours), int(minutes), int(seconds), int(milliseconds))
            times.append(time)
        success, frame = video.read()
        i = i + 1
    video.release()

    # 返回图渠道的图片和相应的时间
    return video_images, times


# 识别读入的图片
# 参数：传入一个图片
# 返回值：如果有车牌 返回车牌
#       如果没有车牌 输出“未检测到车牌”

def identity(image):
    # 对车牌进行定位
    # 返回可能存在车牌的区域集
    plates = plate_Location.plate_locate1(image)

    car_id = '未检测到车牌'
    # 是否识别出车牌
    is_checked = False

    # 读取可能存在车牌的区域
    for plate in plates:
        # 车牌字符分割
        split_char = Split_Char(plate)
        # 分割字符 读取颜色
        char_images, color = split_char.split_and_save_imagesAndColor()

        # 当读取绿色车牌的字符不满8个， 其他车牌的字符不满7个，返回
        if (color == 'green' and len(char_images) < 8) or (color != 'green' and len(char_images) < 7):
            # 如果达不到条件,继续
            continue
        else:
            # 达到条件了就正常输出
            is_checked = True
            break

    # 当车牌满足条件
    if is_checked:
        # 读取车牌和颜色
        string0 = cnn_predict.getstring(color)
        # print(string0)
        car_id = string0
    else:
        # 如果没有读出车牌 输出“未检测到车牌”
        print(car_id)

    return car_id


def video_identity(path):
    # if __name__ == '__main__':
    # video_path = "vedio/plate_vedio.mp4"
    # 读取视频路径
    video_path = path
    # 读取视频名称
    video_name = os.path.basename(video_path)
    # 按帧分割视频
    # 返回值：安贞分割出的图像和相应的时间
    images, times = get_video_seperated(video_path)

    plate_identify_list = []
    # plate_identify_list.append(["video_name", "plate_name","time"])
    # 读取到视频中的车牌信息汇总
    # 文件格式为 [[video_name], [plate_name],[time]]
    # plate_identify_list = []

    for i in range(len(images)):
        img = images[i]
        time = times[i]
        # 识别车牌
        # 如果存在车牌 返回车牌
        # 如果不存在车牌 返回’未检测到车牌‘
        plate = identity(img)

        if plate != '未检测到车牌':
            str = [[video_name], [plate], [time]]
            plate_identify_list.append(str)
            # print(str)

    #     cv.imshow("", img)
    #     cv.waitKey()
    # cv.destroyAllWindows()
    # 返回视频中的车牌信息汇总
    for i in range(len(plate_identify_list)):
        print(plate_identify_list[i])

    return plate_identify_list


# 查找车牌，即输入一个视频和一个车牌号，查找视频中该车牌是否出现过
# 参数：视频路径，车牌号
# 返回值：如果该车牌出现过，则返回该车牌的信息（视频名称，车牌号，出现时间），，如果没出现过，则返回“未找到车牌”
def find_plate(video_path, plate_name):
    # 调用video_identity函数，先识别视频中所有出现的车牌，得到各个车牌的信息，放入video_identity中
    plate_list = video_identity(video_path)
    result = []  # 存放结果的列表
    is_in_video = False  # 布尔值，表示要找的车牌是否在视频中出现过
    # 遍历车牌信息的列表，对比各个车牌号与要查找的车牌号，如果相等，则将该车牌信息加入结果列表，is_in_video设置为True
    for info in plate_list:
        #if info['plate_name'] == plate_name:
        if info[1] == plate_name:
            result.append(info)
            is_in_video = True

    # 根据is_in_video输出结果，为True则返回结果列表，否则返回"未找到车牌"
    if is_in_video:
        return result
    return "未找到车牌"


path = "video/plate_video.mp4"
#video_identity(path)
print("查找到的车牌为：")
print(find_plate(path, ["晋A9900H"]))
