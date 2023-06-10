import numpy as np
import cv2 as cv
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


# 完成数据的正则化
def normalize_data(data):
    return (data - data.mean()) / data.max()


# 加载图片
# 传入参数：图片途径 图片的宽高
def load_image(image_path, width, height):
    # 灰度
    gray_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    # 调整大小
    resized_image = cv.resize(gray_image, (width, height))
    # 正则化
    normalized_image = normalize_data(resized_image)
    data = []
    # 展平
    data.append(normalized_image.ravel())
    # 返回处理好的图片
    return np.array(data)


### 模型定义开始
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
CLASSIFICATION_COUNT = 34
CHINESE_IMAGE_WIDTH = 24
CHINESE_IMAGE_HEIGHT = 48
CHINESE_CLASSIFICATION_COUNT = 31
ENGLISH_LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z']
CHINESE_LABELS = [
    "川", "鄂", "赣", "甘", "贵", "桂", "黑", "沪", "冀", "津",
    "京", "吉", "辽", "鲁", "蒙", "闽", "宁", "青", "琼", "陕",
    "苏", "晋", "皖", "湘", "新", "豫", "渝", "粤", "云", "藏",
    "浙"]


# 构建独热编码
def onehot_labels(labels):
    onehots = np.zeros((len(labels), CLASSIFICATION_COUNT))
    for i in np.arange(len(labels)):
        onehots[i, labels[i]] = 1
    return onehots


# 设置权重
def weight_variable(shape):
    # 截断的产生正态分布的随机数，即随机数与均值的差值若大于两倍的标准差，则重新生成。
    # 均值默认为0
    # 范围在（-0.2 ~ +0.2）
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 设置偏置，并初始化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 设置卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 设置池化层
def max_pool_2x2(x):
    # 设置池化核为2x2：ksize=[1, 2, 2, 1]
    # 设置池化步长，水平和垂直均为2：strides=[1, 2, 2, 1]
    # 设置池化必要时自动padding：padding='SAME'
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def getstring(color):
    ans = ""
    # x 是用来容纳训练数据样本集的特征矩阵，
    x = tf.placeholder(tf.float32, shape=[None, CHINESE_IMAGE_HEIGHT * CHINESE_IMAGE_WIDTH])
    # y_ 是用来容纳训练数据样本集的标签。
    y_ = tf.placeholder(tf.float32, shape=[None, CHINESE_CLASSIFICATION_COUNT])
    # 改变输入数据的形状，让它可以和卷积核进行卷积。
    x_image = tf.reshape(x, [-1, CHINESE_IMAGE_HEIGHT, CHINESE_IMAGE_WIDTH, 1])

    # cnn第一层，卷积核：5*5，颜色通道：1，共有32个卷积核
    W_conv1 = weight_variable([5, 5, 1, 32])
    # cnn第一层，偏置也是32个
    b_conv1 = bias_variable([32])

    # 加上一个激活函数，增加非线性的变换。此处使用的激活函数是 ReLu 函数
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 20x20
    # 池化层 2*2 步长为1
    h_pool1 = max_pool_2x2(h_conv1)  # 20x20 => 10x10

    # cnn第二层，和第一层的构建基本一样。
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 10x10
    h_pool2 = max_pool_2x2(h_conv2)  # 10x10 => 5x5

    # 全连接神经网络的第一个隐藏层
    # 池化层输出的元素总数为：6(H)*12(W)*64(filters)
    W_fc1 = weight_variable([6 * 12 * 64, 1024])  # 全连接第一个隐藏层神经元1024个
    b_fc1 = bias_variable([1024])
    # 把输入的张量(h_pool2)变形/展平。
    h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 12 * 64])  # 转成1列
    # 一样，按照 wx + b 的线性公式构建函数，然后加上 ReLu 函数增加非线性变化。
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # Affine+ReLU

    # 在全链接层之后，加上Dropout 操作，降低过拟合的概率
    # 定义Dropout的比例
    keep_prob = tf.placeholder(tf.float32)  # 定义Dropout的比例
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 执行dropout

    # 全连接神经网络输出层
    # 设置权重 输入神经元数量 输出标签数量
    W_fc2 = weight_variable([1024, CHINESE_CLASSIFICATION_COUNT])  # 全连接输出为10个
    # 设置偏置
    b_fc2 = bias_variable([CHINESE_CLASSIFICATION_COUNT])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    ### 模型定义结束

    # 中文模型
    CHINESE_MODEL_PATH = "opencv_ml/model/cnn_chs/chs.ckpt"
    sess1 = tf.Session()  # 开启会话
    saver1 = tf.train.Saver()  # 开启和加载模型
    saver1.restore(sess1, CHINESE_MODEL_PATH)  # 恢复模型

    # 中文图片的存放地址
    chinese_image_path = "images/chinese.jpg"
    # 加载图片
    chinese_image = load_image(chinese_image_path, CHINESE_IMAGE_WIDTH, CHINESE_IMAGE_HEIGHT)

    # 调用模型进行识别
    results = sess1.run(y_conv, feed_dict={x: chinese_image, keep_prob: 1.0})
    predict = np.argmax(results[0])
    ans += CHINESE_LABELS[predict]
    tf.reset_default_graph()

    # 中文模型结束

    x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT * IMAGE_WIDTH])
    y_ = tf.placeholder(tf.float32, shape=[None, CLASSIFICATION_COUNT])
    x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接神经网络的第一个隐藏层
    # 池化层输出的元素总数为：5(H)*5(W)*64(filters)
    W_fc1 = weight_variable([5 * 5 * 64, 1024])  # 全连接第一个隐藏层神经元1024个
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])  # 转成1列
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # Affine+ReLU

    keep_prob = tf.placeholder(tf.float32)  # 定义Dropout的比例
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 执行dropout

    # 全连接神经网络输出层
    W_fc2 = weight_variable([1024, CLASSIFICATION_COUNT])  # 全连接输出为10个
    b_fc2 = bias_variable([CLASSIFICATION_COUNT])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    sess1.close()

    ### 模型定义结束

    # 英文模型
    ENGLISH_MODEL_PATH = "opencv_ml/model/cnn_enu/enu.ckpt"
    sess = tf.Session()  # 开启会话
    saver = tf.train.Saver()  # 开启和加载模型
    saver.restore(sess, ENGLISH_MODEL_PATH)  # 恢复模型

    platenumber = 6

    for i in range(0, platenumber):
        # 英文字符图片操作
        digit_image_path = "images/english_" + str(i) + ".jpg"
        digit_image = load_image(digit_image_path, IMAGE_WIDTH, IMAGE_HEIGHT)
        # 调用英文模型识别英文字符
        results = sess.run(y_conv, feed_dict={x: digit_image, keep_prob: 1.0})
        predict = np.argmax(results[0])
        ans += ENGLISH_LABELS[predict]

    sess.close()

    # 释放全局状态：这有助于避免旧模型和层造成混乱
    tf.keras.backend.clear_session()

    # 返回识别出的车牌
    return ans
