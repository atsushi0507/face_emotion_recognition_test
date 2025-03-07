from feat import Detector
from feat.utils.io import get_test_data_path
from feat.plotting import imshow
import numpy as np
import os

detector = Detector()

test_data_dir = get_test_data_path()
single_face_img_path = os.path.join(test_data_dir, "single_face.jpg")

single_face_prediction = detector.detect(single_face_img_path, data_type="image")
print(single_face_prediction.emotions)
