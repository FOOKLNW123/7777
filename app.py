import gradio as gr
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
with gr.Blocks() as demo:
    gr.Markdown("## 🎨 ระบบ AI แปลงภาพจริงเป็นภาพอนิเมะและสเก็ต (ทั้งภาพ)")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="📤 อัปโหลดภาพจริง")
            btn_anime = gr.Button("✨ แปลงเป็นภาพอนิเมะ (ทั้งภาพ)")
            btn_sketch = gr.Button("🖊️ แปลงเป็นภาพสเก็ต")
        with gr.Column():
            output_img = gr.Image(label="📥 ผลลัพธ์ที่แปลงแล้ว")

    btn_anime.click(fn=to_anime_full, inputs=input_img, outputs=output_img)
    btn_sketch.click(fn=to_sketch, inputs=input_img, outputs=output_img)

demo.launch()
