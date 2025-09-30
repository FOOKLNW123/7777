import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2

# โหลดโมเดล AnimeGANv2
@st.cache_resource
def load_model():
    model = torch.hub.load(
        "bryandlee/animegan2-pytorch:main",
        "generator",
        pretrained="face_paint_512_v2",
        device="cpu"
    )
    model.eval()
    return model

model = load_model()

# ฟังก์ชันแปลงภาพเป็นอนิเมะ
def to_anime_full(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (512, 512))
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        out = model(img)[0]
    out = (out.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

# ฟังก์ชันแปลงภาพเป็นสเก็ตช์
def to_sketch(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return Image.fromarray(sketch)

# UI
st.title("🎨 WebEase: แปลงภาพเป็นอนิเมะและสเก็ตช์")

uploaded_file = st.file_uploader("📤 อัปโหลดภาพ", type=["jpg", "png"])
style = st.radio("เลือกสไตล์ที่ต้องการ", ["Anime", "Sketch"])

if uploaded_file:
    input_img = Image.open(uploaded_file)
    st.image(input_img, caption="ภาพต้นฉบับ", use_column_width=True)

    if st.button("แปลงภาพ"):
        if style == "Anime":
            output_img = to_anime_full(input_img)
        else:
            output_img = to_sketch(input_img)

        st.image(output_img, caption="ภาพที่แปลงแล้ว", use_column_width=True)
