import os
import uuid
import torch
from diffusers import StableDiffusionPipeline

# Lazy-loaded pipeline (loads only on first use, not at import time)
_pipe = None

def _get_pipeline():
    """
    Load Stable Diffusion pipeline once and reuse.
    Uses float16 on GPU for speed, float32 on CPU as fallback.
    """
    global _pipe

    if _pipe is None:
        print("Loading Stable Diffusion pipeline...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            _pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,   # Uses less VRAM on GPU
                safety_checker=None          # Disable for fashion use
            ).to(device)
        else:
            # CPU fallback (slower, but works without GPU)
            _pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,
                safety_checker=None
            ).to(device)

        print(f"Pipeline loaded on {device} ✅")

    return _pipe


def generate_image(prompt: str) -> str:
    """
    Generates an image from the given prompt using Stable Diffusion.
    Returns the file path of the saved image.
    """
    try:
        pipe = _get_pipeline()

        result = pipe(
            prompt,
            guidance_scale=7.5,
            num_inference_steps=30,     # 30 steps = good quality + reasonable speed
            height=512,
            width=512
        )

        image = result.images[0]

        os.makedirs("outputs", exist_ok=True)
        filename = f"outputs/{uuid.uuid4()}.png"
        image.save(filename)

        return filename

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise RuntimeError("GPU out of memory. Try restarting the server or use a smaller image.")
        raise
