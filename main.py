import os
from dotenv import load_dotenv
import json
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

# LangChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

# Diffusers / SDXL
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image, ImageDraw, ImageFont

load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['HF_HOME'] = "F:\\huggingface_cache"     # Set the Hugging Face cache directory to my F: drive

CACHE_DIR = "F:\\huggingface_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

DEVICE = "cpu"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

llm = ChatGroq(model="llama3-70b-instruct", temperature=0.2)

model_id = "segmind/Segmind-Vega"
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id, torch_dtype=torch.float32, cache_dir=CACHE_DIR, local_files_only=True
)
pipe = pipe.to(DEVICE)

app = FastAPI(title="LangChain + SDXL Digital Documents Generator")

class DocFields(BaseModel):
    name: str = Field(..., description="Recipient full name")
    course: str = Field(..., description="Course or program completed")
    date: str = Field(..., description="Date string; normalize to ISO YYYY-MM-DD")
    issuer: Optional[str] = Field(default="AI Academy", description="Issuing body")
    role: Optional[str] = Field(default="Certificate of Completion", description="Document type/title")
    template_theme: Optional[str] = Field(default="elegant classic", description="Theme or vibe")
    brand_colors: Optional[str] = Field(default="gold, ivory, charcoal", description="Comma list of colors")
    extras: Optional[str] = Field(default="", description="Optional add-ons: QR, signature, logo path")

    @field_validator("date")
    def normalize_date(cls, v):
        # accept "Aug 3", "August 3, 2025", etc. -> ISO if possible
        try:
            # attempt multiple formats
            for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%b %d", "%B %d"):
                try:
                    dt = datetime.strptime(v, fmt)
                    # if year missing, assume current year
                    if "%Y" not in fmt:
                        dt = dt.replace(year=datetime.now().year)
                    return dt.strftime("%Y-%m-%d")
                except:
                    pass
            return datetime.fromisoformat(v).strftime("%Y-%m-%d")
        except:
            return 0
        

parser_prompt = PromptTemplate.from_template(
    """You are a strict information extractor. Given raw user text or JSON, extract clean fields for a certificate.

Return a STRICT JSON object with keys:
"name", "course", "date", "issuer", "role", "template_theme", "brand_colors", "extras"

If any field is missing, infer sensibly (issuer defaults "AI Academy", role defaults "Certificate of Completion").
Prefer concise values. Dates should be human-friendly but short. Avoid extra commentary.

Input:
{raw}
"""
)
parser_chain = parser_prompt | llm | JsonOutputParser()

style_prompt = PromptTemplate.from_template(
    """You are an art director for Stable Diffusion. Create a vivid prompt to generate an ornate background
for a professional certificate.

Context JSON:
{context}

Return ONLY a single line prompt, no JSON, no quotes. The image should be:
- elegant, tasteful, high-contrast text-safe center area
- symmetric ornamental borders or subtle guilloché pattern
- incorporate these colors if provided
- background should not overpower text
- prefer premium print aesthetic (vector-like clarity)

Include style hints (e.g., 'ornate border, filigree, subtle pattern, premium paper texture, studio lighting').
"""
)
style_chain = style_prompt | llm

def generate_bg_image(prompt: str, width=1240, height=1754, num_inference_steps=6, guidance_scale=0.0):
    """
    Generate a background via SDXL. Defaults tuned for speed.
    """
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
    ).images[0]
    return image

def draw_certificate(bg: Image.Image, fields: DocFields) -> Image.Image:
    """
    Lay out text nicely on top of the generated background using PIL.
    """
    img = bg.convert("RGBA")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    # Load fonts (put your .ttf in fonts/)
    def font(size, name="EBGaramond-Regular.ttf"):
        path = os.path.join("fonts", name)
        if not os.path.exists(path):
            # fallback to default PIL font
            return ImageFont.load_default()
        return ImageFont.truetype(path, size=size)

    title_f = font(88)
    name_f  = font(72)
    body_f  = font(40)
    small_f = font(30)

    # Soft panel for text legibility
    panel_margin = int(0.08 * W)
    panel_bbox = [panel_margin, int(0.18*H), W - panel_margin, int(0.82*H)]
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rounded_rectangle(panel_bbox, radius=40, fill=(255,255,255,200))
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    # Centered helpers
    def center_text(s, y, f):
        w, h = draw.textbbox((0,0), s, font=f)[2:]
        draw.text(((W - w)//2, y), s, font=f, fill=(25,25,25,255))
        return h

    y = int(0.23 * H)
    y += center_text(fields.role, y, title_f) + 30
    y += center_text("is proudly presented to", y, small_f) + 20
    y += center_text(fields.name, y, name_f) + 20
    y += center_text("for successfully completing", y, small_f) + 10
    y += center_text(fields.course, y, body_f) + 30
    y += center_text(f"Issued by {fields.issuer} on {fields.date}", y, small_f)

    return img.convert("RGB")

def safe_filename(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in ("-","_")).rstrip()

@app.post("/generate/from-form")
async def generate_from_form(
    name: str = Form(...),
    course: str = Form(...),
    date: str = Form(...),
    issuer: str = Form("AI Academy"),
    role: str = Form("Certificate of Completion"),
    template_theme: str = Form("elegant classic"),
    brand_colors: str = Form("gold, ivory, charcoal"),
    extras: str = Form(""),
):
    # 1. Parse/validate with LangChain parser (normalizes fields if user typed odd strings)
    raw = json.dumps({
        "name": name, "course": course, "date": date,
        "issuer": issuer, "role": role,
        "template_theme": template_theme, "brand_colors": brand_colors, "extras": extras
    })
    parsed = await parser_chain.ainvoke({"raw": raw})
    fields = DocFields(**parsed)

    # 2) Get style prompt from Art Director
    style = await style_chain.ainvoke({"context": json.dumps(parsed, ensure_ascii=False)})
    prompt = f"{style.strip()}, print-ready, center area clean for text"

    # 3) Generate background
    bg = generate_bg_image(prompt)

    # 4) Compose certificate
    out = draw_certificate(bg, fields)

    # 5) Save
    fname = f"certificate_{safe_filename(fields.name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    out_path = os.path.join(OUTPUT_DIR, fname)
    out.save(out_path, "PNG")

    return JSONResponse({"file": f"/download/{fname}", "fields": fields.model_dump(), "prompt_used": prompt})

@app.get("/download/{filename}")
def download(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "file not found"}, status_code=404)
    return FileResponse(path, media_type="image/png", filename=filename)



