from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import base64

from services.gemini_service import get_saree_description
from services.prompt_service import build_prompt
from services.diffusion_service import generate_image

app = FastAPI(title="Saree AI API")

# Allow Android app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Saree AI API is running ✅"}

@app.post("/generate-saree")
async def generate_saree(
    saree_image: UploadFile = File(...),
    hairstyle: str = Form(...),
    lighting: str = Form(...)
):
    temp_path = f"temp_{uuid.uuid4()}.png"

    try:
        # Step 1: Save uploaded image temporarily
        contents = await saree_image.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded image is empty")

        with open(temp_path, "wb") as f:
            f.write(contents)

        # Step 2: Get Gemini description of the saree
        saree_desc = get_saree_description(temp_path)

        # Step 3: Build the image generation prompt
        prompt = build_prompt(saree_desc, hairstyle, lighting)

        # Step 4: Generate image using Stable Diffusion
        output_path = generate_image(prompt)

        # Step 5: Convert output image to base64 so Android can display it
        with open(output_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        # Cleanup generated image file
        os.remove(output_path)

        return {
            "status": "success",
            "saree_description": saree_desc,
            "prompt": prompt,
            "image_base64": image_base64   # Android will decode this
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    finally:
        # Always clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
