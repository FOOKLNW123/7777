
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

# โหลดโมเดล AnimeGANv2 (สไตล์ face_paint_512_v2)
anime_model = torch.hub.load('bryandlee/animegan2-pytorch', 'generator', pretrained='face_paint_512_v2')
anime_model.eval()

# ฟังก์ชัน sharpen ภาพ
def sharpen(img: Image.Image) -> Image.Image:
    img_np = np.array(img)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(img_np, -1, kernel)
    return Image.fromarray(sharpened)

# ฟังก์ชันแปลงภาพทั้งภาพเป็นอนิเมะ
def to_anime_full(img: Image.Image) -> Image.Image:
    original_size = img.size
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output_tensor = anime_model(input_tensor)[0]
    output_tensor = (output_tensor * 0.5 + 0.5).clamp(0, 1)
    output_img = transforms.ToPILImage()(output_tensor)
    return sharpen(output_img.resize(original_size))

# ฟังก์ชันแปลงภาพเป็นสเก็ต
def to_sketch(img: Image.Image) -> Image.Image:
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    inv_img = 255 - img_gray
    blur_img = cv2.GaussianBlur(inv_img, (21, 21), sigmaX=0, sigmaY=0)
    sketch = cv2.divide(img_gray, 255 - blur_img, scale=256)
    return Image.fromarray(sketch)

# สร้าง UI ด้วย Gradio

import streamlit as st
from PIL import Image

st.title("WebEase: แปลงภาพเป็นอนิเมะและสเก็ตช์")

uploaded_file = st.file_uploader("อัปโหลดภาพ", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="ภาพต้นฉบับ", use_column_width=True)

    # ใส่โค้ดแปลงภาพที่นี่


    btn_anime.click(fn=to_anime_full, inputs=input_img, outputs=output_img)
    btn_sketch.click(fn=to_sketch, inputs=input_img, outputs=output_img)

demo.launch()
