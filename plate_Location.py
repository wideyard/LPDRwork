from signal import signal
import cv2 as cv
import numpy as np
import imutils
from scipy import signal

# 灰度值拉伸
def grayStretch(grayImg):
    maxValue = float(grayImg.max())
    minValue = float(grayImg.min())
    for i in range(grayImg.shape[0]):
        for j in range(grayImg.shape[1]):
            grayImg[i, j] = (grayImg[i, j] - minValue) * 255 / (maxValue - minValue)
    return grayImg

# 显示图片
def showImg(img, title="", waitKey=0):
    cv.imshow(title, img)
    cv.waitKey(waitKey)
    cv.destroyAllWindows()

# 预处理
def preprocess(img):
    img = imutils.resize(img, width=640)    # 调整为640宽度
    # showImg(img, "原图", 0)
    grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)   # 转换为灰度图
    grayImg = grayStretch(grayImg)  # 图像灰度拉伸
    # showImg(grayImg, "灰度图", 0)
    blurredImg = cv.GaussianBlur(grayImg, (5, 5), 0)  # 高斯模糊
    # showImg(blurredImg, "高斯模糊", 0)
    gradX = cv.Sobel(blurredImg, cv.CV_16S, 1, 0, ksize=3)
    abs_grad_x = cv.convertScaleAbs(gradX)
    edgedImage = cv.addWeighted(abs_grad_x, 1, 0, 0, 0)
    # showImg(edgedImage, "边缘检测", 0)
    edgedImage = cv.bilateralFilter(edgedImage, 9, 75, 75)
    # showImg(edgedImage, "中值滤波", 0)
    return img, edgedImage

# 找轮廓
def findContours(edgedImage):
    ret, thresholdImage = cv.threshold(edgedImage, 127, 255, cv.THRESH_OTSU) # 二值化
    # showImg(thresholdImage, "thresholdImage")
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (25,3))
    morphologyImage = cv.morphologyEx(thresholdImage, cv.MORPH_CLOSE, kernel)  # 闭运算，连接块
    # showImg(morphologyImage, "morphologyImage")
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,7))
    morphologyImage = cv.morphologyEx(morphologyImage, cv.MORPH_OPEN, kernel)    # 开运算，去除小的凸起
    # showImg(morphologyImage, "morphologyImage")
    contours, _ = cv.findContours(morphologyImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)   # 查找轮廓
    return contours


# 通过形状数据判断是否是车牌
def verify_plate_by_size(contour):
    rect = cv.minAreaRect(contour)
    w, h = rect[1]
    w = int(w)
    h = int(h)
    minRectArea = w * h
    if w==0 or h==0:
        return False
    ratio = w / h
    if ratio < 1:
        ratio = 1 / ratio
    MIN_AREA = 44 * 14
    if minRectArea < MIN_AREA:    # 判断轮廓面积
        return False
    if ratio < 2 or ratio > 6.5:  # 车牌的宽高比在2到6.5之间
        return False
    if rect[2] < 75 and rect[2] > 15: # 倾斜角度过大则不识别
        return False
    return True

def rotate_plate_image(contour, plate_image):
    # 获取该等值线框对应的外接正交矩形(长和宽分别与水平和竖直方向平行)
    x, y, w, h = cv.boundingRect(contour)
    bounding_image = plate_image[y : y + h, x : x + w]
    rect = cv.minAreaRect(contour)      # rect结构： (center (x,y), (width, h eight), angle of rotation)
    rect_width, rect_height = np.int0(rect[1])      # 转成整数
    theta = rect[2]
    direction = 1  # 1表示逆时针，-1表示顺时针
    if rect_width < rect_height:
        # showImg(bounding_image, "angle: " + str(theta) + "width: " + str(rect_width) + "height: " + str(rect_height))
        theta = 90 - theta
        direction = -1
        temp = rect_height
        rect_height = rect_width
        rect_width = temp

    if rect[2] == 90:
        theta = 0
    # 创建一个放大的图像，以便存放之前图像旋转后的结果
    enlarged_width = w * 3 // 2
    enlarged_height = h * 3 // 2
    enlarged_image = np.zeros((enlarged_height, enlarged_width, plate_image.shape[2]), dtype=plate_image.dtype)
    x_in_enlarged = (enlarged_width - w) // 2
    y_in_enlarged = (enlarged_height - h) // 2
    roi_image = enlarged_image[y_in_enlarged:y_in_enlarged+h, x_in_enlarged:x_in_enlarged+w]
    # 将旋转前的图像拷贝到放大图像的中心位置，注意，为了图像完整性，应拷贝boundingRect中的内容
    cv.addWeighted(roi_image, 0, bounding_image, 1, 0, roi_image)
    new_center = (enlarged_width // 2, enlarged_height // 2)
    transform_matrix = cv.getRotationMatrix2D(new_center, theta * direction, 1.0)   # 角度为正，表明逆时针旋转
    transformed_image = cv.warpAffine(enlarged_image, transform_matrix, (enlarged_width, enlarged_height))
    # 截取与最初等值线框长、宽相同的部分
    output_image = cv.getRectSubPix(transformed_image, (rect_width, rect_height), new_center)
    output_image = imutils.resize(output_image, width=440)
    # showImg(output_image, "rotatePlateImage")
    return output_image

def verify_plate_by_VerticalProjection(img):
    grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grayImg = cv.bilateralFilter(grayImg, 5, 50, 50)
    thresholdImg = cv.threshold(grayImg, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    # showImg(thresholdImg, "thresholdImg")
    h, w = thresholdImg.shape[:2]
    projection = h - thresholdImg.sum(axis=0) / 255
    projection_inv = projection * -1
    # 平滑化
    projection_smooth = signal.savgol_filter(projection, 11, 3)
    projection_inv_smooth = signal.savgol_filter(projection_inv, 11, 3)

    peaksIdx, peaksHeight = signal.find_peaks(projection_smooth, height=40, distance = 45)
    peaksIdx_inv, peaksHeight_inv = signal.find_peaks(projection_inv_smooth, height=-20, distance = 45)
    if len(peaksIdx) < 7 - 2 or len(peaksIdx) > 7 + 2:
        return False
    if len(peaksIdx_inv) < 7 - 2 or len(peaksIdx_inv) > 7 + 2:
        return False
    return True        

# 车牌定位
def plate_locate(platePath):
    # print(platePath)
    originImg = cv.imread(platePath)
    # cv.imshow("", originImg)
    originImg, edgedImg = preprocess(originImg)
    contours = findContours(edgedImg)
    plates = []
    for contour in contours:
        if verify_plate_by_size(contour):
            outputImg = rotate_plate_image(contour, originImg)
            if(verify_plate_by_VerticalProjection(outputImg)):
                plates.append(outputImg)
    return plates

def plate_locate1(plate_image):
    originImg = plate_image
    originImg, edgedImg = preprocess(originImg)
    contours = findContours(edgedImg)
    plates = []
    for contour in contours:
        if verify_plate_by_size(contour):
            # tmp = cv.drawContours(originImg.copy(), [contour], -1, (0, 0, 255), 2)
            # showImg(tmp, "verifyPlateBySize")
            outputImg = rotate_plate_image(contour, originImg)
            if(verify_plate_by_VerticalProjection(outputImg)):
                plates.append(outputImg)
    return plates

# basePath = 'G:/code/pro/CCPD2019.tar/platePath = './images/img_1.png'
# plates = plateLocate(platePath)
# for plate in plates:
#     showImg(plate, "plate")
#     cv.waitKey(0)
#     cv.destroyAllWindows()CCPD2019/ccpd_base/'
# imgName = '01-90_85-274&482_457&544-456&545_270&533_261&470_447&482-0_0_16_24_31_28_31-146-27.jpg'
# imgName = '01-90_87-245&456_432&532-445&528_267&529_251&471_429&470-0_0_10_4_28_24_32-54-11.jpg'
# platePath = basePath + imgName
# platePath = 'images/plate2.jpg'
# platePath = './images/img_1.png'
# plates = plateLocate(platePath)
# for plate in plates:
#     showImg(plate, "plate")
#     cv.waitKey(0)
#     cv.destroyAllWindows()