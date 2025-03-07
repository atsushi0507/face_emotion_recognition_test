#ライブラリのインポート
import streamlit as st
import numpy as np
import cv2
import numpy as np
from PIL import Image
st.write(scipy.__version__)
from feat import Detector
import scipy

st.write(scipy.__version__)
st.write(feat.__version__)

@st.cache_resource
def load_detector():
    return Detector()

detector = load_detector()
st.title("Face emotion app")

img_source = st.radio(
    "画像のソースを選択してください。",
    ("画像をアップロード", "カメラで撮影")
)
if img_source == "画像をアップロード":
    img_file_buffer = st.file_uploader("ファイルを選択")
elif img_source == "カメラで撮影":
    img_file_buffer = st.camera_input("カメラで撮影")
else:
    pass

if img_file_buffer :
    img_file_buffer_2 = Image.open(img_file_buffer)
    img_file = np.array(img_file_buffer_2)
    tmp_image = cv2.imwrite('temporary.jpg', img_file)

    image_prediction = detector.detect("temporary.jpg", data_type="image")

    emotion = image_prediction.emotions.idxmax(axis = 1)[0]

    st.markdown("#### あなたの表情は")
    st.markdown(f"### {emotion}です")

    figs = image_prediction.plot_detections(poses=True)

    for fig in figs:
        st.pyplot(fig)