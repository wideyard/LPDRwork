import numpy as np
import cv2 as cv
from plate_locate import util


def locate(path):
    HSV_MIN_BLUE_H = 100					# HSV中蓝色分量最小范围值
    HSV_MAX_BLUE_H = 140					# HSV中蓝色分量最大范围值
    MAX_SV = 255
    MIN_SV = 95

    plate_file_path = path
    plate_image = cv.imread(plate_file_path)

    hsv_image = cv.cvtColor(plate_image, cv.COLOR_BGR2HSV)
    h_split, s_split, v_split = cv.split(hsv_image)				# 将H,S,V分量分别放到三个数组中
    rows, cols = h_split.shape

    binary_image = np.zeros((rows, cols), dtype=np.uint8)

    # 将满足蓝色背景的区域，对应索引的颜色值置为255，其余置为0，从而实现二值化
    for row in np.arange(rows):
        for col in np.arange(cols):
            H = h_split[row, col]
            S = s_split[row, col]
            V = v_split[row, col]
            # 在蓝色值域区间，且满足S和V的一定条件
            if (H >= HSV_MIN_BLUE_H and H<=HSV_MAX_BLUE_H)  \
                and (S >= MIN_SV and S <= MAX_SV)    \
                and (V >= MIN_SV and V <= MAX_SV):
                binary_image[row, col] = 255
            else:
                binary_image[row, col] = 0
    '''
    cv.imshow("",binary_image)
    cv.waitKey(0)
    '''


#    # 执行闭操作，使相邻区域连成一片
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 3))
    morphology_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(morphology_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    verified_plates = []
    for i in np.arange(len(contours)):
        if util.verify_plate_sizes(contours[i]):
            output_image = util.rotate_plate_image(contours[i], plate_image)
            output_image = util.unify_plate_image(output_image)
            verified_plates.append(output_image)
#    #print(len(contours))
#    #cv.imshow("", morphology_image)
#    #cv.waitKey(0)

    for i in np.arange(len(verified_plates)):
        '''
        cv.imshow("", verified_plates[i])
        cv.waitKey()
        '''

        cv.imwrite('result' + str(i) + '.jpg', verified_plates[i])

        gray_image = cv.imread('result' + str(i) + '.jpg', cv.IMREAD_GRAYSCALE)

        # 先利用二值化去除图片噪声
        ret, gray_image = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY)
        '''
        cv.imshow("", gray_image)
        cv.waitKey()
        '''
#        #判定是否为车牌（轮廓有7个），减少误判的可能性
        right = 1
        contours, _ = cv.findContours(gray_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours)<7:
            right = 0
        if (right):
            cv.imwrite('result.jpg', verified_plates[i])

    cv.destroyAllWindows()

#locate("images2/plate1.jpg")