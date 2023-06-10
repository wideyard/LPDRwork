from plate_locate import plate_locaor
import numpy as np
from opencv_char_seperator import plate_char_seperator
import cv2




test_image=cv2.imread("images/0.jpg")
image_name = "015-89_95-115&458_361&524-356&522_104&530_115&454_367&446-0_0_19_24_26_24_20-158-43.jpg"
TRAIN_DIR = 'data/CCPD2019/ccpd_base'


hsv_candidate=plate_locaor.get_candidate_paltes_by_hsv(test_image)
sobel_candidate=plate_locaor.get_candidate_paltes_by_sobel(test_image)



for i in np.arange(len(hsv_candidate)):
    img=hsv_candidate[i]
    cv2.imshow("hsv", img)
    cv2.waitKey()
    candidate_chars = plate_char_seperator.get_candidate_char(img)
    for char in candidate_chars:
        cv2.imshow("", char)
        cv2.waitKey()

# for i in np.arange(len(sobel_candidate)):
#     img=sobel_candidate[i]
#     candidate_chars=plate_char_seperator.get_candidate_char(img)
#     for char in candidate_chars:
#         cv2.imshow("", char)
#         cv2.waitKey()
# cv2.imshow("1",hsv_plate)
# cv2.imshow("2",processing_image)

# cv2.waitKey()
cv2.destroyAllWindows()

