"""
Quick diagnostic: tests image generation with the models available on this API key.
Run: python diagnose.py
"""
import os
from google import genai
from google.genai import types

API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY is not set.")
    exit(1)

client = genai.Client(api_key=API_KEY)
PROMPT = "a simple red apple on a white wooden table, photorealistic"

# ── 1. Imagen 4 via generate_images ─────────────────────────────────────────
print("=== Testing imagen-4.0-generate-001 ===")
try:
    resp = client.models.generate_images(
        model="imagen-4.0-generate-001",
        prompt=PROMPT,
        config=types.GenerateImagesConfig(number_of_images=1),
    )
    if resp.generated_images:
        with open("test_imagen4.png", "wb") as f:
            f.write(resp.generated_images[0].image.image_data)
        print("  SUCCESS — saved test_imagen4.png")
    else:
        print("  FAIL — no images returned")
except Exception as e:
    print(f"  FAIL — {type(e).__name__}: {e}")

# ── 2. gemini-2.5-flash-image via generateContent ───────────────────────────
print("\n=== Testing gemini-2.5-flash-image ===")
try:
    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=PROMPT,
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )
    saved = False
    for part in resp.candidates[0].content.parts:
        if part.inline_data is not None:
            with open("test_flash_image.png", "wb") as f:
                f.write(part.inline_data.data)
            print("  SUCCESS — saved test_flash_image.png")
            saved = True
            break
    if not saved:
        print("  FAIL — no image parts in response")
except Exception as e:
    print(f"  FAIL — {type(e).__name__}: {e}")

# ── 3. gemini-3.1-flash-image-preview via generateContent ───────────────────
print("\n=== Testing gemini-3.1-flash-image-preview ===")
try:
    resp = client.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=PROMPT,
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )
    saved = False
    for part in resp.candidates[0].content.parts:
        if part.inline_data is not None:
            with open("test_3_1_flash.png", "wb") as f:
                f.write(part.inline_data.data)
            print("  SUCCESS — saved test_3_1_flash.png")
            saved = True
            break
    if not saved:
        print("  FAIL — no image parts in response")
except Exception as e:
    print(f"  FAIL — {type(e).__name__}: {e}")
