import gradio as gr
import torch
import numpy as np
from PIL import Image

from diffusers import DDIMScheduler
from pipeline import DynamicIDStableDiffusionPipeline


# =========================
# 1. CẤU HÌNH
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"[INFO] Device: {DEVICE}")


# =========================
# 2. LOAD PIPELINE (THEO PAPER)
# =========================
scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

pipe = DynamicIDStableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    scheduler=scheduler,
    torch_dtype=DTYPE,
)

pipe = pipe.to(DEVICE)

print("[OK] DynamicID Pipeline loaded successfully")


# =========================
# 3. HÀM TIỀN XỬ LÝ ẢNH
# =========================
def preprocess_face(image: Image.Image):
    """
    Chuẩn paper:
    - Resize
    - Normalize
    - Không dùng image như img2img
    """
    image = image.convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image.to(DEVICE, dtype=DTYPE)


# =========================
# 4. HÀM SINH ẢNH (PIPELINE ĐÚNG PAPER)
# =========================
@torch.no_grad()
def generate(face_image, prompt):
    if face_image is None:
        return None

    # ---- Face Encoder ----
    face_tensor = preprocess_face(face_image)

    # Pipeline nội bộ sẽ thực hiện:
    # face_tensor
    # → Face Encoder
    # → IMR
    # → DiSeNet
    # → ReENet
    # → Inject UNet Cross-Attention

    result = pipe(
        prompt=prompt,
        face_image=face_tensor,     # ❗ đúng paper
        num_inference_steps=30,
        guidance_scale=7.5,
        height=512,
        width=512,
    )

    return result.images[0]


# =========================
# 5. GIAO DIỆN GRADIO
# =========================
with gr.Blocks() as demo:
    gr.Markdown("## 🧬 DynamicID – Paper-Accurate Demo")

    with gr.Row():
        with gr.Column():
            input_face = gr.Image(
                label="Input Face (Identity Source)",
                type="pil"
            )

            prompt = gr.Textbox(
                label="Text Prompt",
                value="a professional portrait photo, high quality"
            )

            btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output = gr.Image(label="Generated Image")

    btn.click(
        fn=generate,
        inputs=[input_face, prompt],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
