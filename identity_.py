from flask import Flask, url_for
from flask import render_template, request
import os
from flask import send_from_directory
from werkzeug.utils import secure_filename
import plate_Location
import cv2 as cv
from split import Split_Char
import cnn_predict
from pathlib import Path

app = Flask(__name__)
# 进行相关的配置文件
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = os.getcwd()+'\\static\\upload\\'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB


# 定义验证后缀的脚本文件
def allowed_file(filename):
    print(filename.rsplit('.', 1)[1])
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def logo():
    return render_template('logo.html')


@app.route('/main.html',methods=['GET', 'POST'])
def identity():
    #设置初始化
    chinese = "default.png"
    english1 = "default.png"
    english2 = "default.png"
    english3 = "default.png"
    english4 = "default.png"
    english5 = "default.png"
    english6 = "default.png"
    english7 = "default.png"


    if request.method == 'POST':
        file = request.files['uploadfile']
        print(file.filename)
        if file and allowed_file(file.filename):
            # 将初始的给保存一下
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # 然后对图片进行大小的处理
            imagePath = app.config['UPLOAD_FOLDER'] + filename
            imagePath = Path(imagePath)
            # 这一块是调用一些接口，然后进行相关的操作,输入的参数是imagePath
            # 将方法的返回值赋给car_id
            car_id = 666
            plates = plateLocation.plateLocate(str(imagePath))
            # print(1111)
            # print(len(plates))
            # for i in range(0, len(plates)):
            #     cv.imwrite("/debugger/plate"+str(i)+".jpg", plates[i])
            #     print(plates[i])
            car_id = '识别出错'
            is_checked = False
            plates = plateLocation.plateLocate(str(imagePath))
            for plate in plates:
                split_char = Split_Char(plate)
                char_images, color = split_char.split_and_save_imagesAndColor()
                if (color == 'green' and len(char_images) < 8) or (color != 'green' and len(char_images) < 7):
                    # 如果达不到条件,继续
                    continue
                else:
                    # 达到条件了就正常输出
                    is_checked = True
                    break

            if is_checked == True:
                string0 = cnn_predict.getstring(color)
                print(string0)
                car_id = string0

                chinese = "chinese.jpg"
                english1 = "english_0.jpg"
                english2 = "english_1.jpg"
                english3 = "english_2.jpg"
                english4 = "english_3.jpg"
                english5 = "english_4.jpg"
                english6 = "english_5.jpg"
                if color =="green":
                    english7 = "english_6.jpg"
                else:
                    english7 ="default.png"
                return render_template('main.html', imagename=filename, plate="origin.jpg", msg=car_id +"  "+color,
                                       chinese=chinese, english1=english1, english2=english2, english3=english3,
                                       english4=english4, english5=english5, english6=english6, english7=english7)


            file_url = url_for('identity', filename=filename)
            return render_template('main.html', imagename=filename, plate="default.jpg", msg=car_id,
                                   chinese=chinese, english1=english1, english2=english2, english3=english3,
                                   english4=english4, english5=english5, english6=english6, english7=english7)
        else:
            msg = '错误，上传文件类型错误'
            return render_template('main.html', msg=msg)
    return render_template('main.html', msg="", plate="default.png", imagename="default.png", chinese=chinese, english1=english1, english2=english2, english3=english3, english4=english4, english5=english5, english6=english6, english7=english7)


if __name__ == '__main__':
    # 调用Flask实例对象run()方法启动Flask工程
    app.run()
