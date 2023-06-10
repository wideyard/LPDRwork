import numpy as np
import cv2 as cv
# import tensorflow as tf
# import tensorflow.compat.v1 as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()
from opencv_char_seperator import plate_char_seperator
from plate_locate import plate_locaor
import os
import hsv

### 模型定义开始
ENGLISH_MODEL_PATH = "opencv_ml/model/cnn_enu/enu.ckpt"
CHINESE_MODEL_PATH = "opencv_ml/model/cnn_chs/chs.ckpt"
ENGLISH_IMAGE_WIDTH = 20
ENGLISH_IMAGE_HEIGHT = 20
CHINESE_IMAGE_WIDTH = 24
CHINESE_IMAGE_HEIGHT = 48

ENGLISH_LABELS = [
	'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
	'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',	'J', 'K',
	'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
	'W', 'X', 'Y', 'Z']

CHINESE_LABELS = [
	"川","鄂","赣","甘","贵","桂","黑","沪","冀","津",
	"京","吉","辽","鲁","蒙","闽","宁","青","琼","陕",
	"苏","晋","皖","湘","新","豫","渝","粤","云","藏",
	"浙"]

LABEL_DICT = {
	'chuan':0, 'e':1, 'gan':2, 'gan1':3, 'gui':4, 'gui1':5, 'hei':6, 'hu':7, 'ji':8, 'jin':9,
	'jing':10, 'jl':11, 'liao':12, 'lu':13, 'meng':14, 'min':15, 'ning':16, 'qing':17,	'qiong':18, 'shan':19,
	'su':20, 'sx':21, 'wan':22, 'xiang':23, 'xin':24, 'yu':25, 'yu1':26, 'yue':27, 'yun':28, 'zang':29,
	'zhe':30
}

CHINESE_CLASS_COUNT = 31
ENGLISH_CLASS_COUNT = 34


def normalize_data(data):
    return (data - data.mean()) / data.max()

def load_image(image, width, height):
    #gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_image = image
    #gray_image = cv.imread(image, cv.IMREAD_GRAYSCALE)
    #gray_image = cv.convert()

    # 先利用二值化去除图片噪声
    ret, img = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY)

    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    n = len(contours)  # 轮廓的个数
    cv_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area <= 10:
            cv_contours.append(contour)
        else:
            continue

    cv.fillPoly(img, cv_contours, (255, 255, 255))
    cv.imshow("",img)
    cv.waitKey(0)

    resized_image = cv.resize(gray_image, (width, height))
    normalized_image = normalize_data(resized_image)
    data = []
    data.append(normalized_image.ravel())
    return np.array(data)


def onehot_labels(labels, CLASSIFICATION_COUNT):
    onehots = np.zeros((len(labels), CLASSIFICATION_COUNT))
    for i in np.arange(len(labels)):
        onehots[i, labels[i]] = 1
    return onehots

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # padding='SAME',使卷积输出的尺寸=ceil(输入尺寸/stride)，必要时自动padding
    # padding='VALID',不会自动padding，对于输入图像右边和下边多余的元素，直接丢弃
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def enu_model(IMAGE_HEIGHT, IMAGE_WIDTH, CLASSIFICATION_COUNT):
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT * IMAGE_WIDTH])
    y_ = tf.placeholder(tf.float32, shape=[None, CLASSIFICATION_COUNT])
    x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])  # color channel == 1; 32 filters
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 20x20
    h_pool1 = max_pool_2x2(h_conv1)  # 20x20 => 10x10

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 10x10
    h_pool2 = max_pool_2x2(h_conv2)  # 10x10 => 5x5

    # 全连接神经网络的第一个隐藏层
    # 池化层输出的元素总数为：5(H)*5(W)*64(filters)
    W_fc1 = weight_variable([5 * 5 * 64, 1024])  # 全连接第一个隐藏层神经元1024个
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])  # 转成1列
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # Affine+ReLU

    # keep_prob = tf.placeholder(tf.float32)                          # 定义Dropout的比例
    rate = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, rate=rate)  # keep_prob)                    # 执行dropout

    # 全连接神经网络输出层
    W_fc2 = weight_variable([1024, CLASSIFICATION_COUNT])  # 全连接输出为10个
    b_fc2 = bias_variable([CLASSIFICATION_COUNT])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return x,y_conv,rate



### 模型定义结束
def chs_model(IMAGE_HEIGHT, IMAGE_WIDTH, CLASSIFICATION_COUNT):
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT * IMAGE_WIDTH])
    y_ = tf.placeholder(tf.float32, shape=[None, CLASSIFICATION_COUNT])
    x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])  # color channel == 1; 32 filters
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 20x20
    h_pool1 = max_pool_2x2(h_conv1)  # 20x20 => 10x10

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 10x10
    h_pool2 = max_pool_2x2(h_conv2)  # 10x10 => 5x5

    # 全连接神经网络的第一个隐藏层
    # 池化层输出的元素总数为：5(H)*5(W)*64(filters)
    W_fc1 = weight_variable([6 * 12 * 64, 1024])  # 全连接第一个隐藏层神经元1024个
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 12 * 64])  # 转成1列
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # Affine+ReLU

    # keep_prob = tf.placeholder(tf.float32)                          # 定义Dropout的比例
    rate = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, rate=rate)  # keep_prob)                    # 执行dropout

    # 全连接神经网络输出层
    W_fc2 = weight_variable([1024, CLASSIFICATION_COUNT])  # 全连接输出为10个
    b_fc2 = bias_variable([CLASSIFICATION_COUNT])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return x,y_conv,rate

### 模型定义结束

def recognition(plate_image):
    digit_image_path = plate_image
    hsv_candidate = hsv.get_candidate_paltes_by_hsv(plate_image)
    sobel_candidate = plate_locaor.get_candidate_paltes_by_sobel(plate_image)

    if hsv_candidate == False:
        return

    for i in np.arange(len(hsv_candidate)):
        img = hsv_candidate[i]
        cv.imwrite("plate1.jpg", img)
    # img = load_image(plate_image, ENGLISH_IMAGE_WIDTH, ENGLISH_IMAGE_HEIGHT)

    # digit_image = load_image(plate_image, ENGLISH_IMAGE_WIDTH, ENGLISH_IMAGE_HEIGHT)
    # # digit_image = load_image(img, ENGLISH_IMAGE_WIDTH, ENGLISH_IMAGE_HEIGHT)
    # x, y_conv, rate = enu_model(ENGLISH_IMAGE_HEIGHT, ENGLISH_IMAGE_WIDTH, ENGLISH_CLASS_COUNT)
    # # 识别英文
    #
    # sess2 = tf.Session()
    # saver = tf.train.Saver()
    # saver.restore(sess2, ENGLISH_MODEL_PATH)
    # results = sess2.run(y_conv, feed_dict={x: digit_image, rate: 0.0})  # keep_prob: 1.0})
    # predict = np.argmax(results[0])
    # print(ENGLISH_LABELS[predict])

        candidate_chars = plate_char_seperator.get_candidate_char(img)
        count = 0
        for char in candidate_chars:
            cv.imshow("", char)
            cv.waitKey()
            if count == 0:
                g1 = tf.Graph()  # 加载到Session 1的graph
                # 识别中文
                with g1.as_default():
                    #digit_image = load_image(img, CHINESE_IMAGE_WIDTH, CHINESE_IMAGE_HEIGHT)
                    digit_image = load_image(char, CHINESE_IMAGE_WIDTH, CHINESE_IMAGE_HEIGHT)
                    x, y_conv, rate = chs_model(CHINESE_IMAGE_HEIGHT, CHINESE_IMAGE_WIDTH, CHINESE_CLASS_COUNT)

                    sess1 = tf.Session()
                    saver = tf.train.Saver()
                    saver.restore(sess1, CHINESE_MODEL_PATH)

                    results = sess1.run(y_conv, feed_dict={x: digit_image, rate: 0.0})  # keep_prob: 1.0})
                    predict = np.argmax(results[0])

                    print(CHINESE_LABELS[predict])
                    sess1.close()

            else:
                g2 = tf.Graph()  # 加载到Session 2的graph
                with g2.as_default():
                    # digit_image = char
                    digit_image = load_image(char, ENGLISH_IMAGE_WIDTH, ENGLISH_IMAGE_HEIGHT)
                    # digit_image = load_image(img, ENGLISH_IMAGE_WIDTH, ENGLISH_IMAGE_HEIGHT)
                    x, y_conv, rate = enu_model(ENGLISH_IMAGE_HEIGHT, ENGLISH_IMAGE_WIDTH, ENGLISH_CLASS_COUNT)
                    # 识别英文

                    sess2 = tf.Session()
                    saver = tf.train.Saver()
                    saver.restore(sess2, ENGLISH_MODEL_PATH)
                    results = sess2.run(y_conv, feed_dict={x: digit_image, rate: 0.0})  # keep_prob: 1.0})
                    predict = np.argmax(results[0])
                    print(ENGLISH_LABELS[predict])
                    sess2.close()

            count += 1

    cv.destroyAllWindows()


# IMAGE_DIR = "../images"
# image_dir = []
# image_names = []
# for item in os.listdir(IMAGE_DIR):
#     image = cv.imread(os.path.join(IMAGE_DIR, item))
#     cv.imshow("", image)
#     cv.waitKey()
#     cv.destroyAllWindows()
#     recognition(image)


# image = cv.imread("opencv_ml/data/chs_train/sx/282.jpg")
# cv.imshow("first", image)
# cv.waitKey()
# cv.destroyAllWindows()
# recognition(image)




