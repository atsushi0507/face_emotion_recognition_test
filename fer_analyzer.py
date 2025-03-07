from feat import Detector
from feat.utils.io import get_test_data_path
import os
import sys

detector = Detector()

test_data = os.path.join(get_test_data_path(), "single_face.jpg")
pred = detector.detect(test_data, data_type="image")
