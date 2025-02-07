import easyocr
import cv2
import numpy as np
from pathlib import Path
import streamlit as st
import os
import requests
from PIL import Image
import io
import fitz  # PyMuPDF
import json
from groq import Groq
import openai
from openai import OpenAI
import time
import streamlit.components.v1 as components
import modalhtml as mh
from pix2tex.cli import LatexOCR

# 初始化EasyOCR阅读器
reader_bi = easyocr.Reader(['ch_sim', 'en'], gpu=True)
reader_zh = easyocr.Reader(['ch_sim'], gpu=True)
reader_en = easyocr.Reader(['en'], gpu=True)
# 初始化 LatexOCR 模型
math_model = LatexOCR()

# 处理函数
def process_file(uploaded_file, language):
    if uploaded_file is None:
        return ""

    # 处理PDF
    if uploaded_file.type == "application/pdf":
        doc = fitz.open(stream=uploaded_file.read())
        text = ""
        for page in doc:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            result = reader_bi.readtext(img)
            text += "\n".join([res[1] for res in result]) + "\n"
        return text

    # 处理图片
    else:
        try:
            # 获取上传文件的字节数据
            img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            # 使用 cv2.imdecode 解码
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

            # 进行 OCR 识别
            if language == '中文':
                result = reader_zh.readtext(img)
            elif language == '英文':
                result = reader_en.readtext(img)
            elif language == '中英混合':
                result = reader_bi.readtext(img)
            elif language == '数学公式':
                # 将 OpenCV 图像转换为 PIL 图像
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # 使用 pix2tex 进行数学公式识别
                latex_code = math_model(img_pil)
                return latex_code

            # 输出识别结果
            return "\n".join([res[1] for res in result])
        except Exception as e:
            return f"识别过程中出现错误: {e}"
