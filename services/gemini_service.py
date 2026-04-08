import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Load API key from .env file (never hardcode keys!)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

def get_saree_description(image_path: str) -> str:
    """
    Uses Gemini to describe the saree in the given image.
    Returns a detailed text description for use in image generation.
    """
    try:
        img = Image.open(image_path)

        response = model.generate_content([
            (
                "You are a fashion expert. Describe this saree in detail for an AI image generator. "
                "Include: color, fabric type, pattern/design, border style, pallu details, "
                "embroidery or embellishments, and how it is draped. "
                "Be specific and vivid. Keep response under 100 words."
            ),
            img
        ])

        return response.text.strip()

    except Exception as e:
        raise RuntimeError(f"Gemini description failed: {str(e)}")
