import numpy as np
import streamlit as st
from PIL import Image
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# ------------------ App Config ------------------ #
st.set_page_config(page_title="AI Photo Enhancer", page_icon="✨", layout="centered")
st.title("✨ AI Photo Enhancer ")
st.write("Upload a blurry image and enhance it with AI!")

# ------------------ Model Setup ------------------ #
@st.cache_resource  # cache so it loads only once
def load_model():
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23,
        num_grow_ch=32, scale=4
    )
    sr_model = RealESRGANer(
        scale=4,
        model_path="RealESRGAN_x4plus.pth",
        model=model,
        tile=0, tile_pad=0, pre_pad=0,
        half=False,
        device=torch.device('cpu'),
        # 🔥 Add this line to override internal dtype
        model_preload=False  # Prevent auto-loading with half precision
    )
    sr_model.model.load_state_dict(torch.load("RealESRGAN_x4plus.pth", map_location="cpu"))
    sr_model.model.eval()
    return sr_model





sr_model = load_model()
print(f"Model device: {next(sr_model.model.parameters()).device}")
print(f"Model dtype: {next(sr_model.model.parameters()).dtype}")
import os
if not os.path.exists("RealESRGAN_x4plus.pth"):
    st.error("❌ Model file not found! Please place 'RealESRGAN_x4plus.pth' in the root directory.")
# ------------------ File Upload ------------------ #
uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if not uploaded_file:
    st.info("📁 Please upload an image to begin enhancement.")


if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Uploaded Image", use_column_width=True)

    if st.button("🚀 Enhance Image"):
        with st.spinner("Enhancing with RealESRGAN... ⏳"):
            img_np = np.array(img).astype(np.float32)
            output, _ = sr_model.enhance(img_np, outscale=4)
            output_img = Image.fromarray(output)

            # Show result
            st.image(output_img, caption="✨ Enhanced Image", use_column_width=True)

            # Download button
            output_img.save("enhanced.jpg")
            with open("enhanced.jpg", "rb") as file:
                st.download_button(
                    label="⬇️ Download Enhanced Image",
                    data=file,
                    file_name="enhanced.jpg",
                    mime="image/jpeg"
                )