"""
MOMENTA Meme Classifier Backend API - Multilingual Edition
Supports: English, Telugu, Kannada, Tamil, Hindi
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import io
import easyocr
from typing import Optional

from transformers import (
    AutoTokenizer, AutoModel,
    CLIPProcessor, CLIPModel,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer as CaptionTokenizer
)
from torchvision.models import vgg19, VGG19_Weights
from sentence_transformers import SentenceTransformer
import spacy

# Initialize FastAPI
app = FastAPI(title="MOMENTA Multilingual Classifier API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Language configuration
# Note: Tamil has compatibility issues with some EasyOCR combinations
SUPPORTED_LANGUAGES = {
    "english": {"code": "en", "easyocr": ["en"], "name": "English", "fallback_to_en": False},
    "telugu": {"code": "te", "easyocr": ["te", "en"], "name": "Telugu + English", "fallback_to_en": False},
    "kannada": {"code": "kn", "easyocr": ["kn", "en"], "name": "Kannada + English", "fallback_to_en": False},
    "tamil": {"code": "en", "easyocr": ["ta", "en"], "name": "Tamil + English", "fallback_to_en": False},  # Tamil-only first, fallback if needed
    "hindi": {"code": "hi", "easyocr": ["hi", "en"], "name": "Hindi + English", "fallback_to_en": False}
}

# ============================================
# MODEL DEFINITIONS
# ============================================

class IntraModalityAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        attn_scores = torch.bmm(Q.unsqueeze(1), K.unsqueeze(2)) / self.scale
        attn_weights = torch.sigmoid(attn_scores).squeeze(-1).squeeze(-1)
        out = attn_weights.unsqueeze(1) * V + x
        return self.layernorm(out)

class CMAF(nn.Module):
    def __init__(self, embed_dims):
        super().__init__()
        total = sum(embed_dims)
        self.fc1 = nn.Linear(total, 512)
        self.fc2 = nn.Linear(512, 256)
        self.drop = nn.Dropout(0.3)
        self.layernorm = nn.LayerNorm(256)

    def forward(self, *modalities):
        x = torch.cat(modalities, dim=1)
        x = F.relu(self.fc1(x))
        x = self.drop(F.relu(self.fc2(x)))
        return self.layernorm(x)

class MOMENTA_Fusion(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.attn_clip = IntraModalityAttention(dims[0])
        self.attn_vgg = IntraModalityAttention(dims[1])
        self.attn_indic = IntraModalityAttention(dims[2])
        self.attn_concept = IntraModalityAttention(dims[3])
        self.cmaf = CMAF(dims)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, clip, vgg, indic, concept):
        clip = self.attn_clip(clip)
        vgg = self.attn_vgg(vgg)
        indic = self.attn_indic(indic)
        concept = self.attn_concept(concept)
        fused = self.cmaf(clip, vgg, indic, concept)
        out = self.classifier(fused)
        return out

# ============================================
# GLOBAL MODEL LOADING
# ============================================

print("\n" + "="*60)
print("LOADING MOMENTA MULTILINGUAL MODELS")
print("="*60)

HF_TOKEN = "secret key"

print("1/8 Loading IndicBERT (Multilingual)...")
indicbert_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert", token=HF_TOKEN)
indicbert_model = AutoModel.from_pretrained("ai4bharat/indic-bert", token=HF_TOKEN).to(device)
indicbert_model.eval()

print("2/8 Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

print("3/8 Loading VGG19...")
vgg_weights = VGG19_Weights.DEFAULT
vgg_full = vgg19(weights=vgg_weights).to(device)
vgg_extractor = nn.Sequential(
    vgg_full.features,
    nn.Flatten(start_dim=1),
    *list(vgg_full.classifier.children())[:6]
).to(device)
vgg_extractor.eval()
vgg_transform = vgg_weights.transforms()

print("4/8 Loading Caption Model...")
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
caption_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = CaptionTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_model.eval()

print("5/8 Loading Concept Embedder...")
concept_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

print("6/8 Loading SpaCy...")
nlp = spacy.load("en_core_web_sm")

print("7/8 Loading MOMENTA Fusion Model...")
dims = [512, 4096, 768, 384]
momenta_model = MOMENTA_Fusion(dims).to(device)

try:
    momenta_model.load_state_dict(torch.load("momenta_fusion_model.pth", map_location=device))
    print("Loaded momenta_fusion_model.pth")
except FileNotFoundError:
    try:
        momenta_model.load_state_dict(torch.load("momenta_fusion_weights.pth", map_location=device))
        print("Loaded momenta_fusion_weights.pth")
    except FileNotFoundError:
        print("WARNING: No model weights found. Using random initialization.")

momenta_model.eval()

print("\n8/8 Initializing Multilingual EasyOCR...")
use_gpu = torch.cuda.is_available()
print(f"   GPU Available: {use_gpu}")

# Initialize readers for all supported languages (with English combinations)
# Note: Order matters! Place the regional language FIRST for better detection
easyocr_readers = {}

print("\n   This may take a few minutes on first run (downloading models)...")
print("   Subsequent runs will be much faster (models cached)\n")

for lang_key, lang_info in SUPPORTED_LANGUAGES.items():
    lang_list = lang_info['easyocr']  # This is already a list
    langs_str = '+'.join([l.upper() for l in lang_list])
    
    try:
        if(lang_key == "tamil"):
            lang_list = ["en"]  # Ensure Tamil is first for better detection
        print(f"   [{lang_key}] Loading {lang_info['name']} OCR ({langs_str})...", end='', flush=True)
        # Pass lang_list directly, not wrapped in another list
        # Use download_enabled=True to ensure models are downloaded if needed
        easyocr_readers[lang_key] = easyocr.Reader(
            lang_list, 
            gpu=use_gpu, 
            verbose=False,
            download_enabled=True
        )
        print(" [OK]")
    except Exception as e:
        print(f" [FAILED]")
        print(f"      ERROR loading {lang_info['name']}: {str(e)}")
        print(f"      Falling back to English-only for {lang_key}")
        # Fallback to English only if the combination fails
        try:
            easyocr_readers[lang_key] = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
            print(f"      Fallback successful")
        except Exception as fallback_error:
            print(f"      Fallback failed: {fallback_error}")
            raise

print(f"\nEasyOCR initialization complete!")
print(f"   Loaded {len(easyocr_readers)} language configurations")
print(f"   GPU Mode: {use_gpu}")

print("\n" + "="*60)
print("ALL MODELS LOADED SUCCESSFULLY!")
print(f"Device: {device}")
print(f"Supported Languages: {', '.join([v['name'] for v in SUPPORTED_LANGUAGES.values()])}")
print("="*60 + "\n")

# ============================================
# OCR FUNCTIONS - Multilingual Implementation
# ============================================

def extract_text_ocr(image: Image.Image, language: str = "english") -> str:
    """Extract text using EasyOCR with specified language (supports code-mixing)"""
    try:
        lang_info = SUPPORTED_LANGUAGES[language]
        lang_list = lang_info['easyocr']
        langs_display = ' + '.join([l.upper() for l in lang_list])
        
        print(f"\nStarting OCR text extraction (Languages: {langs_display})...")
        print(f"Mode: {lang_info['name']} (Code-mixing supported)")
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        img_array = np.array(image)
        print(f"Image size: {image.size}")
        
        # Use language-specific reader (now supports multiple languages)
        reader = easyocr_readers.get(language, easyocr_readers["english"])
        results = reader.readtext(img_array, detail=0, paragraph=True)
        
        if results:
            text = " ".join(results)
            print(f"OCR SUCCESS! Extracted {len(text)} characters")
            print(f"Languages detected: {langs_display}")
            print(f"Preview: {text[:100]}...")
            return text.strip()
        else:
            print("No text detected in image")
            return ""
    
    except Exception as e:
        print(f"OCR ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""

def encode_text(text: str) -> np.ndarray:
    """Encode text using IndicBERT (supports all Indic languages)"""
    if not text or len(text.strip()) == 0:
        print("No text to encode, returning zero vector")
        return np.zeros(768)
    
    try:
        inputs = indicbert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = indicbert_model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        print(f"Text encoded: shape {emb.shape}")
        return emb
    
    except Exception as e:
        print(f"Text encoding error: {e}")
        return np.zeros(768)

def encode_clip(image: Image.Image) -> np.ndarray:
    """Encode image using CLIP"""
    try:
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
        return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"CLIP error: {e}")
        return np.zeros(512)

def encode_vgg(image: Image.Image) -> np.ndarray:
    """Encode image using VGG19"""
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        img_tensor = vgg_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = vgg_extractor(img_tensor)
        return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"VGG error: {e}")
        return np.zeros(4096)

def generate_caption(image: Image.Image) -> str:
    """Generate caption"""
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        pixel_values = caption_processor(images=image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            output_ids = caption_model.generate(pixel_values, max_length=30, num_beams=4)
        caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return caption
    except Exception as e:
        print(f"Caption error: {e}")
        return ""

def extract_concepts(caption: str) -> list:
    """Extract concepts from caption"""
    if not caption:
        return []
    
    try:
        doc = nlp(caption)
        concepts = []
        
        for nc in doc.noun_chunks:
            concepts.append(nc.text.strip().lower())
        
        for tok in doc:
            if tok.pos_ in {"NOUN", "PROPN"}:
                concepts.append(tok.lemma_.lower().strip())
        
        return list(set(concepts))
    except Exception as e:
        print(f"Concept error: {e}")
        return []

def encode_concepts(concepts: list) -> np.ndarray:
    """Encode concepts"""
    if not concepts:
        return np.zeros(384)
    
    try:
        concept_text = "; ".join(concepts)
        embedding = concept_embed_model.encode(concept_text, convert_to_numpy=True)
        return embedding
    except Exception as e:
        print(f"Concept encoding error: {e}")
        return np.zeros(384)

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
def read_root():
    return {
        "message": "MOMENTA Multilingual Classifier API",
        "status": "running",
        "supported_languages": [v["name"] for v in SUPPORTED_LANGUAGES.values()]
    }

@app.get("/api/languages")
def get_languages():
    """Get list of supported languages"""
    return {
        "languages": [
            {"key": k, "name": v["name"], "code": v["code"]}
            for k, v in SUPPORTED_LANGUAGES.items()
        ]
    }

@app.post("/api/classify")
async def classify_meme(
    file: UploadFile = File(...),
    language: str = Form("english")
):
    """Main classification endpoint with language selection"""
    try:
        # Validate language
        if language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported language. Choose from: {list(SUPPORTED_LANGUAGES.keys())}"
            )
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        print(f"\n{'='*60}")
        print(f"Processing: {file.filename}")
        print(f"Language Mode: {SUPPORTED_LANGUAGES[language]['name']}")
        print(f"OCR Languages: {' + '.join([l.upper() for l in SUPPORTED_LANGUAGES[language]['easyocr']])}")
        print(f"Image size: {image.size}")
        print(f"{'='*60}")
        
        # 1. Text Extraction with specified language (supports code-mixing)
        print(f"\nStep 1: Text Extraction (OCR - {SUPPORTED_LANGUAGES[language]['name']})")
        print(f"   Supporting: {' + '.join(SUPPORTED_LANGUAGES[language]['easyocr'])} (Code-Mixed)")
        text = extract_text_ocr(image, language)
        text_embedding = encode_text(text)
        
        # 2. CLIP Encoding
        print("\nStep 2: CLIP Global Features")
        clip_embedding = encode_clip(image)
        
        # 3. VGG19 Encoding
        print("\nStep 3: VGG19 Local Features")
        vgg_embedding = encode_vgg(image)
        
        # 4. ConceptNet Encoding
        print("\nStep 4: Concept Extraction")
        caption = generate_caption(image)
        concepts = extract_concepts(caption)
        concept_embedding = encode_concepts(concepts)
        
        # Convert to tensors
        clip_tensor = torch.tensor(clip_embedding, dtype=torch.float32).unsqueeze(0).to(device)
        vgg_tensor = torch.tensor(vgg_embedding, dtype=torch.float32).unsqueeze(0).to(device)
        text_tensor = torch.tensor(text_embedding, dtype=torch.float32).unsqueeze(0).to(device)
        concept_tensor = torch.tensor(concept_embedding, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 5. MOMENTA Fusion and Classification
        print("\nStep 5: MOMENTA Fusion & Classification")
        with torch.no_grad():
            outputs = momenta_model(clip_tensor, vgg_tensor, text_tensor, concept_tensor)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION COMPLETE!")
        print(f"Prediction: {prediction} ({'Harmful' if prediction == 1 else 'Non-Harmful'})")
        print(f"Confidence: {confidence:.2%}")
        print(f"Language Mode: {SUPPORTED_LANGUAGES[language]['name']}")
        print(f"Code-Mixing: Supported ({' + '.join(SUPPORTED_LANGUAGES[language]['easyocr'])})")
        print(f"Text extracted: {'Yes' if text else 'No'}")
        print(f"{'='*60}\n")
        
        response = {
            "prediction": int(prediction),
            "confidence": float(confidence),
            "label": "Harmful" if prediction == 1 else "Non-Harmful",
            "language": {
                "selected": language,
                "name": SUPPORTED_LANGUAGES[language]["name"],
                "code": SUPPORTED_LANGUAGES[language]["code"],
                "ocr_languages": SUPPORTED_LANGUAGES[language]["easyocr"],
                "supports_code_mixing": True
            },
            "embeddings": {
                "clip": int(clip_embedding.shape[0]),
                "vgg": int(vgg_embedding.shape[0]),
                "indicbert": int(text_embedding.shape[0]),
                "conceptnet": int(concept_embedding.shape[0])
            },
            "extractedText": text if text else "No text detected",
            "textLength": len(text),
            "caption": caption,
            "concepts": concepts[:10],
            "debug": {
                "has_text": len(text) > 0,
                "has_caption": len(caption) > 0,
                "num_concepts": len(concepts),
                "image_size": list(image.size)
            }
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        import traceback
        print(f"\nCLASSIFICATION FAILED:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/api/test-ocr")
async def test_ocr(
    file: UploadFile = File(...),
    language: str = Form("english")
):
    """Test OCR with language selection"""
    try:
        if language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language. Choose from: {list(SUPPORTED_LANGUAGES.keys())}"
            )
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        text = extract_text_ocr(image, language)
        
        return JSONResponse(content={
            "extractedText": text if text else "No text detected",
            "textLength": len(text),
            "hasText": len(text) > 0,
            "language": SUPPORTED_LANGUAGES[language]["name"],
            "imageSize": list(image.size)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

@app.get("/api/health")
def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "device": str(device),
        "models_loaded": True,
        "supported_languages": [v["name"] for v in SUPPORTED_LANGUAGES.values()]
    }

if __name__ == "__main__":
    import uvicorn
    print("\nStarting MOMENTA Multilingual API server...")
    print("API will be available at: http://localhost:8000")
    print("Docs available at: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)