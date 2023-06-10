import cv2 as cv


def dis_black_white(image):
    """区分黑白,image是二值图"""
    black_count = 0
    white_count = 0
    for x in range(0,image.shape[1]):
        for y in range(0,image.shape[0]):
            current_pixel_color = image[y][x]
            if current_pixel_color<127:
                black_count+=1
            else:
                white_count+=1
    if black_count>white_count:
        return 'black'
    else:
        return 'white'



def split(image,keep_borders=False):
    """
    用于将车牌分割成单个字符
    :param image:车牌对应的矩阵
    :return:一组字符图片
    """
    # 第0步,判断图片规格,将图片规格变为440*140,达到裁边的效果
    image_y = image.shape[0]
    image_x = image.shape[1]
    image = image[5:image_y - 5, 5:image_x - 5]
    cv.imwrite('./images/resized_origin_image.jpg',image)
    if image_x / image_y < 2.5:  # 大车牌
        size = "big"
    else:  # 小车牌
        size = 'small'
    image = cv.resize(image, (440, 140))#统一压缩成小车牌形式,为了之后颜色遍历节约时间
    image_y = image.shape[0]
    image_x = image.shape[1]

    # 第1步,判断车牌类型(车牌类型不同位数也不同)
    color = dis_color(image)
    if color == 'green':#分离各种车牌的汉字,字母/数字组合,认为只有新能源汽车为8位
        num_count = 7
        chinese_count = 1
    elif color!='unknown color':
        num_count = 6
        chinese_count = 1
    else:
        print("颜色识别出错!")
        return None

    # 第2步,将RGB图片降噪后变为灰度图,然后二值化
    image = cv.GaussianBlur(image, (3, 3), 0)  # 高斯滤波
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  # 灰度化
    _, image = cv.threshold(image, 0, 255, cv.THRESH_OTSU)  # 二值化
    if color == "BW":
        color = dis_black_white(image)
    if color != 'blue' and color != 'black':  # 除了蓝色车牌,黑色车牌其他车牌都要反向,变成黑底,白字
        image = cv.bitwise_not(image)

    # showPic(image,0)

    # 第3步,执行开操作,即膨胀+腐蚀获得较好的轮廓
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # 3x3开操作核
    image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)  # 开操作
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    # image = cv.dilate(image, kernel)

    # showPic(image,1)

    # 第4步,查找所有可能的轮廓,转化为rect数组
    try_counts = 10 #试十次
    for i in range(try_counts):
        contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 查找轮廓,外轮廓,仅拐点信息
        rect_list = getRectAreas(contours)  # 转存成rectList形式
        if len(rect_list)<=7:
            image_x, image_y, image = remove_border_image(image)
        else:
            print("原始边框数: ", len(rect_list))
            break

    # 第5步,找出车牌字符矩阵
    standard_ratio = 1 / 2  # width/height的标准值为1/2
    volatility = 1  # 浮动率
    char_images = []
    real_rect = []

    # 专门为yellow上下版面打造
    char_images_up = []
    char_images_down = []
    real_rect_up = []
    real_rect_down = []

    upper_bound = standard_ratio + standard_ratio * volatility
    lower_bound = standard_ratio - standard_ratio * volatility
    for x, y, w, h in rect_list:
        ratio = float(w / h)
        if color == 'yellow' and size == 'big':
            if y < image_y * 1 / 3:  # 分为上下两个部分,上部分由于图片被resize,所以ratio取倒数
                ratio = 1 / ratio
                if ratio > lower_bound and ratio < upper_bound and w > 5 and h > 20 and w < 120 and h < 200:
                    single_char_image = image[y:y + h, x:x + w]
                    char_images_up.append(single_char_image)
                    real_rect_up.append([x, y, w, h])
            else:
                if ratio >= lower_bound and ratio <= upper_bound and w > 5 and h > 30 and w < 100 and h < 200:
                    single_char_image = image[y:y + h, x:x + w]
                    char_images_down.append(single_char_image)
                    real_rect_down.append([x, y, w, h])
        elif size == 'small':
            if h > w and ratio > lower_bound and ratio < upper_bound and w > 10 and h > 50 and w < 100 and h < 200:
                single_char_image = image[y:y + h, x:x + w]
                char_images.append(single_char_image)
                real_rect.append([x, y, w, h])

    if color == 'yellow' and size == 'big':#大车牌的合并操作
        real_rect = real_rect_down.copy()
        if len(real_rect_up) > 0:
            real_rect.insert(0, real_rect_up[-1])
        char_images = char_images_down.copy()
        if len(char_images_up) > 0:
            char_images.insert(0, char_images_up[-1])
    # 第6步,汉字单独处理
    char_images = char_images[-num_count:]  # 字母部分基本没问题
    real_rect = real_rect[-num_count:]
    try:
        chinese_processX(char_images, image, real_rect, color)
        if keep_borders==False:
            chinese_processY(char_images, image, real_rect)
    except:
        print(image, '图像未分割出字符!')
    # (测试)第7步,输出字符照片集,验证
    # showPic(char_images, -1)

    #第8步,回调
    return char_images,color


def remove_border_image(image):
    """重裁边框,返回image_x和image_y"""
    image_y = image.shape[0]
    image_x = image.shape[1]
    y_split_value = 1
    x_split_value = 5
    image = image[y_split_value:image_y - y_split_value, x_split_value:image_x - x_split_value]
    return image_x, image_y, image


def dis_color(image):
    """
    将车牌的总体颜色分为5类: (blue,green,yellow,black,white)
    :param image: 原始图像
    :return:车牌颜色
    """
    blue = green = yellow = black = white = 0
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    start_X = 0
    end_X = image_hsv.shape[0]
    start_Y = 0
    end_Y = image_hsv.shape[1]
    area_of_image = end_Y * end_X

    for x in range(start_X, end_X, 2):  # 跳2步.以进行等距采样,降低时间消耗
        for y in range(start_Y, end_Y, 2):
            H, S, V = image_hsv[x][y]
            if 11 < H <= 34 and S > 34:  # 黄色
                yellow += 1
            elif 35 < H <= 99 and S > 34:  # 绿色
                green += 1
            elif 99 < H <= 124 and S > 34:  # 蓝色
                blue += 1

            if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:  # 黑色
                black += 1
            elif 0 < H < 180 and 0 < S < 43 and 46 < V < 225:  # 白色
                white += 1

    color_weight = 8
    if blue * color_weight >= area_of_image:
        return 'blue'
    if green * color_weight >= area_of_image:
        return 'green'
    if yellow * color_weight >= area_of_image:
        return 'yellow'

    if (black+white) * color_weight * 4 >= area_of_image:
        #判定为黑白图片后使用RGB灰度图来精确判定
        return "BW"
    return "BW"


def chinese_processX(char_images, image, real_rect, color, bias=0):
    """
    单独处理汉字
    :param char_images: 已经处理好的字母集
    :param image: 整体图片
    :param real_rect:已经处理好的矩阵
    :param bias:x方向上偏置,默认为0
    """
    start_X = 0 + bias
    end_X = real_rect[0][0] - 5
    start_Y = max(real_rect[0][1] - 5, 0)
    end_Y = min(start_Y + real_rect[0][3] + 10, image.shape[0])

    # showPic(image)

    chinese_x = start_X
    chinese_y = start_Y
    chinese_x_end = start_X
    chinese_y_end = end_Y
    efficient_ratio_limit = 0.1  # 有效占比
    give_up_count_limit = 10  # 10个像素没有达到可行比就结束循环
    give_up_count = 0
    is_start_scan = False  # 开始进入扫描了吗
    for x in range(start_X, end_X):
        sum_y_pixel = 0  # 这一列总的像素值
        efficient_y_pixel = 0  # 不同颜色的像素值
        for y in range(start_Y, end_Y):
            if image[y][x] == 255:
                efficient_y_pixel += 1
            sum_y_pixel += 1
        efficient_ratio = efficient_y_pixel / sum_y_pixel

        if efficient_ratio < efficient_ratio_limit and is_start_scan == True:  # 不达标 且 已经开始扫描
            give_up_count += 1
            chinese_x_end = x
        if efficient_ratio < efficient_ratio_limit and is_start_scan == False:  # 不达标 且 未开始扫描
            chinese_x = chinese_x_end = x
        if efficient_ratio > efficient_ratio_limit and is_start_scan == True:  # 达标 且 已开始扫描
            give_up_count = 0
            chinese_x_end = x
        if efficient_ratio > efficient_ratio_limit and is_start_scan == False:  # 达标 且 未开始扫描
            is_start_scan = True
            chinese_x = chinese_x_end = x
        if give_up_count >= give_up_count_limit and is_start_scan == True:
            break

    # 进行修正,因为初始的偏旁部首可能一开始像素并未达标
    chinese_x -= 3
    if chinese_x < 0:
        chinese_x = 0
    chinese_width = chinese_x_end - chinese_x
    chinese_height = chinese_y_end - chinese_y
    # 进行校验,看长度是不是差不多
    max_width = 0
    for item in real_rect:
        if item[2] > max_width:
            max_width = item[2]
    if color == 'yellow':  # 黄色车牌是竖着的,取第一个一样尺寸的即可
        max_width = real_rect[0][2]
    bias_ratio = 0.7  # 允许存在的误差

    if chinese_width > max_width * bias_ratio and chinese_width < max_width / bias_ratio:
        # 如果符合则插入
        chinese_rect = [chinese_x, chinese_y, chinese_width, chinese_height]
        real_rect.insert(0, chinese_rect)
        char_images.insert(0, image[chinese_y:chinese_y_end, chinese_x:chinese_x_end])
    elif chinese_x_end < end_X:
        bias += chinese_x_end  # 下一次迭代的初始偏置,用于跳过图像一开始的不明污点
        chinese_processX(char_images, image, real_rect, color, bias)
    return None


def chinese_processY(char_images, image, real_rect):
    """
    对y方向进行裁边
    :param char_images: 7张字符图像
    :param real_rect: 7张字符图像对应的矩形
    :return:None
    """
    chinese_image = char_images[0]

    start_X = 0
    start_Y = 0
    end_X = real_rect[0][2]
    end_Y = real_rect[0][3]

    # 要收紧的上下界
    upper_bound_y_bias = 0
    lower_bound_y_bias = 0

    efficient_ratio_limit = 0.1  # 有效占比
    for y in range(start_Y, end_Y):#从上到下裁边
        sum_x_pixel = 0  # 这一行总的像素值
        efficient_x_pixel = 0  # 不同颜色的像素值
        for x in range(start_X, end_X):
            if chinese_image[y][x] == 255:
                efficient_x_pixel += 1
            sum_x_pixel += 1
        efficient_ratio = efficient_x_pixel / sum_x_pixel
        if efficient_ratio > efficient_ratio_limit:
            break
        else:
            upper_bound_y_bias += 1

    for y in range(end_Y - 1, start_Y - 1, -1):#从下到上裁边
        sum_x_pixel = 0  # 这一行总的像素值
        efficient_x_pixel = 0  # 不同颜色的像素值
        for x in range(start_X, end_X):
            if chinese_image[y][x] == 255:
                efficient_x_pixel += 1
            sum_x_pixel += 1
        efficient_ratio = efficient_x_pixel / sum_x_pixel
        if efficient_ratio > efficient_ratio_limit:
            break
        else:
            lower_bound_y_bias += 1

    # 重定义区域
    real_rect[0][1] += upper_bound_y_bias
    real_rect[0][3] -= (lower_bound_y_bias + upper_bound_y_bias)
    char_images[0] = image[real_rect[0][1]:real_rect[0][1] + real_rect[0][3],
                     real_rect[0][0]:real_rect[0][0] + real_rect[0][2]]

    return None


def getRectAreas(contours):
    """
    输入一组边框数组,得到每个边框矩形化后的数组(排序后)
    :param contours: 原始边框数组
    :return:rectList:矩形参数的边框数组
    """
    rectList = []
    for item in contours:
        rect = cv.boundingRect(item)
        rectList.append(rect)
    rectList = sorted(rectList, key=lambda s: s[0], reverse=False)  # 根据左上角排序(x)
    return rectList


# 以下为调试方法----------------------------------------------------------------------------------------------------
def showPic(image, windowNum=0):
    if windowNum != -1:
        cv.imshow(str(windowNum + 100), image)
        cv.waitKey()
    else:
        for i in range(len(image)):
            cv.imshow(str(i), image[i])
        cv.waitKey()
#-----------------------------------------------------------------------------------------------------------------
class Split_Char():
    def __init__(self, image, keep_borders=False, path = None):
        """
        :param image: 原始图像
        :param keep_borders: 是否保留上下边框
        :param path:通过文件路径读取图片
        """
        self.image = None
        self.path = None
        if path is not None:
            try:
                self.path = path
                self.image = cv.imread(path)
            except:
                print("路径错误,未找到文件")
        else:
            self.image = image
        self.keep_borders = keep_borders
        self.char_images = []
        self.color = None
    def put_image(self,image, keep_borders=False, path = None):
        """单独放入文件,将重置所有参数"""
        self.image = None
        self.path = None
        if path is not None:
            try:
                self.path = path
                self.image = cv.imread(path)
            except:
                print("路径错误,未找到文件")
        else:
            self.image = image
        self.keep_borders = keep_borders
        self.char_images = []
        self.color = None

    def put_split_return_imagesAndColor(self,image,keep_borders = False):
        """放入一张图片并返回分割后的图片和颜色"""
        self.image = image
        self.keep_borders = keep_borders
        self.char_images = []
        self.color = None
        self.path = None
        self.split_image()
        return self.char_images,self.color

    def put_split_return_imagesAndColor_by_path(self,path,keep_borders = False):
        """通过路径放入图片,返回分割后的图片和颜色"""
        try:
            self.path = path
            self.image = cv.imread(path)
        except:
            print("路径错误,未找到文件")
        self.image = cv.imread(path)
        self.keep_borders = keep_borders
        self.char_images = []
        self.color = None
        self.split_image()
        return self.char_images, self.color

    def split_image(self):
        """
        处理图片,调用这个主方法
        """
        self.char_images, self.color = split(image=self.image, keep_borders=self.keep_borders)

    def get_split_images(self):
        """获得分割后的照片"""
        return self.char_images

    def split_and_return_imagesAndColor(self):
        """处理并返回分割后的字符"""
        self.split_image()
        return self.get_split_images(),self.color
    def split_and_save_imagesAndColor(self):
        """保存到images目录下"""
        IMAGE_WIDTH = 20
        IMAGE_HEIGHT = 20
        CHINESE_IMAGE_WIDTH = 24
        CHINESE_IMAGE_HEIGHT = 48
        self.split_image()
        save_path = './images/'
        origin_save_path = save_path+'origin.jpg'
        cv.imwrite(origin_save_path, self.image)
        chinese_save_path = save_path+'chinese.jpg'
        self.char_images[0] = cv.resize(self.char_images[0],(CHINESE_IMAGE_WIDTH,CHINESE_IMAGE_HEIGHT))
        # self.char_images[0] = cv.resize(self.char_images[0],(IMAGE_WIDTH,IMAGE_HEIGHT))
        cv.imwrite(chinese_save_path, self.char_images[0])

        #这里是马士祺又做了一个文件夹需要放静态资源
        save_path_mpp = 'static/images/'
        origin_save_path_mpp = save_path_mpp+'origin.jpg'
        cv.imwrite(origin_save_path_mpp, self.image)
        chinese_save_path_mpp = save_path_mpp + 'chinese.jpg'
        cv.imwrite(chinese_save_path_mpp,self.char_images[0])

        for i in range(1, len(self.char_images)):
            english_save_path = save_path+'english_'+str(i-1)+'.jpg'
            self.char_images[i] = cv.resize(self.char_images[i], (IMAGE_WIDTH-8, IMAGE_HEIGHT))
            self.char_images[i] = cv.copyMakeBorder(self.char_images[i], 0, 0, 4, 4, cv.BORDER_CONSTANT, value=[0, 0, 0])
            # self.char_images[i] = cv.resize(self.char_images[i], (CHINESE_IMAGE_WIDTH,CHINESE_IMAGE_HEIGHT))
            cv.imwrite(english_save_path,self.char_images[i])

            #这里也是马士祺保存文件
            english_save_path_mpp = save_path_mpp + 'english_' + str(i - 1) + '.jpg'
            cv.imwrite(english_save_path_mpp, self.char_images[i])

        color_save_path = save_path+'color.txt'
        file = open(color_save_path,'w')
        file.write(self.color)

        return self.get_split_images(), self.color

    def get_color(self):
        """字符串类型,有blue,green,yellow,black,white (黑白) 5种"""
        return self.color

    def show_split_image(self):
        """展示分割后的照片"""
        showPic(self.char_images,-1)

    def show_origin_image(self):
        """展示原图"""
        showPic(self.image, 278)


if __name__ == '__main__':
    path = './images/img_29.png'
    image = cv.imread(path)
    split(image)
