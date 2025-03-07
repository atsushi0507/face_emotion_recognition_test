#ライブラリのインポート
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from PIL import Image


#py-featでモデルを構築する。色々いじる余地があるので公式ドキュメント参照
from feat import Detector
detector = Detector(

)
from feat.utils.io import get_test_data_path


#本編  
st.title("Face emotion app")

#画像アップロードかカメラを選択するようにする。
img_source = st.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "カメラで撮影":
  img_file_buffer = st.camera_input("カメラで撮影")
elif img_source == "画像をアップロード":
  img_file_buffer = st.file_uploader("ファイルを選択")
else:
    pass


#どちらを選択しても後続の処理は同じ
if img_file_buffer :
  img_file_buffer_2 = Image.open(img_file_buffer)
  img_file = np.array(img_file_buffer_2)
  cv2.imwrite('temporary.jpg', img_file)
  
  #py-featの表情解析結果をデータフレーム形式でimage_predictionとする
  image_prediction = detector.detect_image("temporary.jpg")
  
  #感情に関するカラムだけを残す
  image_prediction = image_prediction[["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]]
  
  #感情の最大値を示すデータのカラム名（感情名）をemotionとする
  emotion = image_prediction.idxmax(axis = 1)[0]

  figs = image_prediction.plot_detections(poses=True)

  st.markdown("#### あなたの表情は")
  st.markdown(f"### {emotion}です")

  for i in range(figs):
     st.pyplot(figs[i])