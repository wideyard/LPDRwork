import cv2
import numpy as np
import matplotlib.pyplot as plt
import plate_locate.util
import locate_hsv_plate

def imread_photo(filename, flags=cv2.IMREAD_COLOR):
    """
    该函数能够读取磁盘中的图片文件，默认以彩色图像的方式进行读取
    输入： filename 指的图像文件名（可以包括路径）
          flags用来表示按照什么方式读取图片，有以下选择（默认采用彩色图像的方式）：
              IMREAD_COLOR 彩色图像
              IMREAD_GRAYSCALE 灰度图像
              IMREAD_ANYCOLOR 任意图像
    输出: 返回图片的通道矩阵
    """
    return cv2.imread(filename, flags)

def resize_photo(imgArr,MAX_WIDTH = 1000):
    """
    这个函数的作用就是来调整图像的尺寸大小，当输入图像尺寸的宽度大于阈值（默认1000），我们会将图像按比例缩小
    输入： imgArr是输入的图像数字矩阵
    输出:  经过调整后的图像数字矩阵
    拓展：OpenCV自带的cv2.resize()函数可以实现放大与缩小，函数声明如下：
            cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) → dst
        其参数解释如下：
            src 输入图像矩阵
            dsize 二元元祖（宽，高），即输出图像的大小
            dst 输出图像矩阵
            fx 在水平方向上缩放比例，默认值为0
            fy 在垂直方向上缩放比例，默认值为0
            interpolation 插值法，如INTER_NEAREST，INTER_LINEAR，INTER_AREA，INTER_CUBIC，INTER_LANCZOS4等
    """
    img = imgArr
    rows, cols= img.shape[:2]     #获取输入图像的高和宽
    if cols >  MAX_WIDTH:
        change_rate = MAX_WIDTH / cols
        img = cv2.resize(img ,( MAX_WIDTH ,int(rows * change_rate) ), interpolation = cv2.INTER_AREA)
    return img

def predict(imageArr):
    """
    这个函数通过一系列的处理，找到可能是车牌的一些矩形区域
    输入： imageArr是原始图像的数字矩阵
    输出：gray_img_原始图像经过高斯平滑后的二值图
          contours是找到的多个轮廓
    """
    img_copy = imageArr.copy()
    gray_img = cv2.cvtColor(img_copy , cv2.COLOR_BGR2GRAY)
    gray_img_ = cv2.GaussianBlur(gray_img, (5,5), 0, 0, cv2.BORDER_DEFAULT)
    kernel = np.ones((23, 23), np.uint8)
    img_opening = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    img_opening = cv2.addWeighted(gray_img, 1, img_opening, -1, 0)
    # 找到图像边缘
    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_edge = cv2.Canny(img_thresh, 100, 200)
    # # 使用开运算和闭运算让图像边缘成为一个整体
    kernel = np.ones((10, 10), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    # # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return gray_img_,contours


#分割！！！！！！！！！！！！！！！！！！！！！！！！！！！

#1.对待分割的车牌图片进行预处理：灰度处理+去除边缘

# 根据设定的阈值和图片直方图，找出波峰，用于分隔字符；返回波峰值
def find_waves(threshold, histogram):
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


def remove_plate_upanddown_border(card_img, clear):
    """
    这个函数将截取到的车牌照片转化为灰度图，然后去除车牌的上下无用的边缘部分，确定上下边框
    输入： card_img是从原始图片中分割出的车牌照片
    输出: 在高度上缩小后的字符二值图片
    """
    plate_Arr = cv2.imread(card_img)
    plate_gray_Arr = cv2.cvtColor(plate_Arr, cv2.COLOR_BGR2GRAY) #将车牌图像转换为灰度图；cvtColor()：颜色空间转换函数；COLOR_BGR2GRAY：转换类型的整数代码 BGR->灰度
    ret, plate_binary_img = cv2.threshold(plate_gray_Arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #Otsu图像阈值处理

    #print(clear)
    if(clear):
        # 去掉外部白色边框，以免查找轮廓时仅找到外框
        offsetX = 5
        offsetY = 5
        offset_region = plate_binary_img[offsetY:-offsetY, offsetX - 3:-offsetX]
        plate_binary_img = np.copy(offset_region)
        #print("yes")


    row_histogram = np.sum(plate_binary_img, axis=1)  # 数组的每一行求和
    row_min = np.min(row_histogram)
    row_average = np.sum(row_histogram) / plate_binary_img.shape[0]
    row_threshold = (row_min + row_average) / 2
    wave_peaks = find_waves(row_threshold, row_histogram)   #返回波峰点
    # 接下来挑选跨度最大的波峰
    wave_span = 0.0
    for wave_peak in wave_peaks:
        span = wave_peak[1] - wave_peak[0]
        if span > wave_span:
            wave_span = span
            selected_wave = wave_peak
    plate_binary_img = plate_binary_img[selected_wave[0]:selected_wave[1], :]
    '''
    cv2.imshow("plate_binary_img", plate_binary_img)
    cv2.waitKey(0)
    '''


    # 先利用二值化去除图片噪声
    ret, plate_binary_img = cv2.threshold(plate_binary_img, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(plate_binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)  # 轮廓的个数
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 5:
            cv_contours.append(contour)
        else:
            continue

    cv2.fillPoly(plate_binary_img, cv_contours, (0, 0, 0))
    '''
    cv2.imshow("", plate_binary_img)
    cv2.waitKey(0)
    '''


    return plate_binary_img

    ##################################################
    # 测试用
    #print( row_histogram )
    #fig = plt.figure()
    #plt.hist( row_histogram )
    #plt.show()
    # 其中row_histogram是一个列表，列表当中的每一个元素是车牌二值图像每一行的灰度值之和，列表的长度等于二值图像的高度
    # 认为在高度方向，跨度最大的波峰为车牌区域
    #cv2.imshow("plate_gray_Arr", plate_binary_img[selected_wave[0]:selected_wave[1], :])
    ##################################################



#2.进行分割

#####################二分-K均值聚类算法############################

def distEclud(vecA, vecB):
    """
    计算两个坐标向量之间的曼哈顿距离
    """
    return np.sum(abs(vecA - vecB))


def randCent(dataSet, k):  #质心
    n = dataSet.shape[1]  # 列数
    centroids = np.zeros((k, n))  # 用来保存k个类的质心
    for j in range(n):
        minJ = np.min(dataSet[:, j], axis=0)    #每列的min
        rangeJ = float(np.max(dataSet[:, j])) - minJ
        for i in range(k):
            centroids[i:, j] = minJ + rangeJ * (i + 1) / k
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = dataSet.shape[0]   #行数
    clusterAssment = np.zeros((m, 2))  # 这个簇分配结果矩阵包含两列，一列记录簇索引值，第二列存储误差。这里的误差是指当前点到簇质心的街区距离
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0] == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    """
    这个函数首先将所有点作为一个簇，然后将该簇一分为二。之后选择其中一个簇继续进行划分，选择哪一个簇进行划分取决于对其划分是否可以最大程度降低SSE的值。
    输入：dataSet是一个ndarray形式的输入数据集
          k是用户指定的聚类后的簇的数目
         distMeas是距离计算函数
    输出:  centList是一个包含类质心的列表，其中有k个元素，每个元素是一个元组形式的质心坐标
            clusterAssment是一个数组，第一列对应输入数据集中的每一行样本属于哪个簇，第二列是该样本点与所属簇质心的距离
    """
    m = dataSet.shape[0]  #dataSet的行数，即plate_binary_Arr的大于等于255的元素数
    clusterAssment = np.zeros((m, 2))    #生成m行2列的全0数组
    centroid0 = np.mean(dataSet, axis=0).tolist()   #每列的平均值
    centList = []
    centList.append(centroid0)
    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.array(centroid0), dataSet[j, :]) ** 2   #计算距离
    while len(centList) < k:  # 小于K个簇时
        lowestSSE = np.inf   #正无穷，无具体数值
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0] == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)   #将簇一分为二
            sseSplit = np.sum(splitClustAss[:, 1])
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0] != i), 1])
            if (sseSplit + sseNotSplit) < lowestSSE:  # 如果满足，则保存本次划分
                bestCentTosplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 0)[0], 0] = bestCentTosplit
        centList[bestCentTosplit] = bestNewCents[0, :].tolist()
        centList.append(bestNewCents[1, :].tolist())
        clusterAssment[np.nonzero(clusterAssment[:, 0] == bestCentTosplit)[0], :] = bestClustAss
    return centList, clusterAssment


def split_licensePlate_character(plate_binary_img):
    """
    此函数用来对车牌的二值图进行水平方向的切分，将字符分割出来
    输入： plate_gray_Arr是车牌的二值图，rows * cols的数组形式
    输出： character_list是由分割后的车牌单个字符图像二值图矩阵组成的列表
    """
    plate_binary_Arr = np.array(plate_binary_img)
    row_list, col_list = np.nonzero(plate_binary_Arr >= 255)  #选出plate_binary_Arr矩阵中的大于等于255的非0元素的位置（行列号）
    dataArr = np.column_stack((col_list, row_list))  # 将列索引和行索引合并成一个数组。dataArr的第一列是列索引，第二列是行索引，要注意
    centroids, clusterAssment = biKmeans(dataArr, 7, distMeas=distEclud)   #分成7部分
    centroids_sorted = sorted(centroids, key=lambda centroid: centroid[0])   #lambda函数，参数为centroid，输出为centroid[0]
    split_list = []
    for centroids_ in centroids_sorted:
        i = centroids.index(centroids_)   #第一个匹配项的索引位置
        current_class = dataArr[np.nonzero(clusterAssment[:, 0] == i)[0], :]
        x_min, y_min = np.min(current_class, axis=0)
        x_max, y_max = np.max(current_class, axis=0)
        split_list.append([y_min, y_max, x_min, x_max])
    character_list = []
    for i in range(len(split_list)):
        single_character_Arr = plate_binary_img[split_list[i][0]: split_list[i][1], split_list[i][2]:split_list[i][3]]
        character_list.append(single_character_Arr)
        cv2.imshow('character' + str(i), single_character_Arr)
        cv2.imwrite('mypic'+str(i)+'.jpg',single_character_Arr)
        #plt.imshow('character' + str(i), single_character_Arr)
        #plt.savefig('myfig.png')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #print(col_histogram)
    # fig = plt.figure()
    # plt.hist( col_histogram )
    # plt.show()

    return character_list ,len(split_list) # character_list中保存着每个字符的二值图数据

    ############################
    # 测试用
    # print(col_histogram )
    # fig = plt.figure()
    # plt.hist( col_histogram )
    # plt.show()
    ############################





# if __name__ == "__main__":
    # locate_hsv_plate.locate("images2/candidate_plate.jpg")#("images/plate1.jpg")
    #测试分割字符
    # split_licensePlate_character(remove_plate_upanddown_border("result.jpg", True))


    #test = cv2.imread("data/enu_test/enu_test/0/210.jpg")
    #img = np.array(img1)
    #print(type(img))
    #print(type(test))


    # 测试remove，得出处理后的车牌图片
    #remove_plate_upanddown_border("images/candidate_plate.jpg")

    #img = imread_photo("images/plate2.jpg")
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray_image = resize_photo(gray_img,MAX_WIDTH = 1000)
    ##gray_img_,contours = predict(img)
    #cv2.imshow('img', img)
    ##cv2.imshow('gray_img_', gray_img_)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #print(row_histogram)
    # fig = plt.figure()
    # plt.hist( row_histogram )
    # plt.show()
    # 其中row_histogram是一个列表，列表当中的每一个元素是车牌二值图像每一行的灰度值之和，列表的长度等于二值图像的高度
    # 认为在高度方向，跨度最大的波峰为车牌区域
    # cv2.imshow("plate_gray_Arr", plate_binary_img[selected_wave[0]:selected_wave[1], :])