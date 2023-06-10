import plate_Location
from split import Split_Char
import cnn_predict

def identity_function(imagePath):
    car_id = '未识别到车牌'
    is_checked = False
    plates = plate_Location.plate_locate(str(imagePath))
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
        return car_id
    else:
        print(car_id)
        return car_id

