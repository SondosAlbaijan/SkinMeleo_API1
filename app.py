import os, io, base64
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import tensorflow as tf
from tensorflow.keras.models import load_model

from PIL import Image, ImageOps

# Enable AVIF support if plugin exists
try:
    import pillow_avif
except Exception:
    pass



# CONFIG

# Image input size
IMG_SIZE = (240, 240)

# Output class labels
CLASS_NAMES = ["MEL","NV","BCC","AKIEC","BKL","DF","VASC"]

# Threshold for mole detection
MOLE_THRESHOLD = 0.99

# Threshold for abnormal classification
ABNORMAL_THRESHOLD = 0.50

# API authentication key
API_KEY = ""

# Model paths
KERAS_PRIMARY  = os.getenv("KERAS_PRIMARY", "models/api_export.keras")
KERAS_FALLBACK = os.getenv("KERAS_FALLBACK", "models/SkinMeleo_model.keras")


# PREPROCESS

def preprocess_rgb(file_bytes: bytes) -> np.ndarray:
    # Load image from bytes
    img = Image.open(io.BytesIO(file_bytes))

    # Fix image orientation and convert to RGB
    img = ImageOps.exif_transpose(img).convert("RGB")

    # Resize image to model input size
    img = img.resize(IMG_SIZE, resample=Image.NEAREST)

    # Convert image to NumPy array
    arr = np.asarray(img, dtype=np.float32)

    # Apply EfficientNet preprocessing
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)

    # Add batch dimension
    return np.expand_dims(arr, 0)



# AUTH

def authorize(auth_header: str | None):
    # Skip authorization if API key is not set
    if not API_KEY:
        return

    # Validate Authorization header format
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing/invalid Authorization header")

    # Extract token
    token = auth_header.split(" ", 1)[1].strip()

    # Validate token
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# LOAD MODEL

# Select available model path
KERAS_PATH = KERAS_PRIMARY if os.path.exists(KERAS_PRIMARY) else KERAS_FALLBACK

# Ensure model file exists
if not os.path.exists(KERAS_PATH):
    raise RuntimeError("Model file not found in models/")

# Load trained model
model = load_model(KERAS_PATH, compile=False)

# Warm-up model
_ = model(tf.zeros((1,) + IMG_SIZE + (3,)))

# Print model status
print("✅ Loaded model from:", KERAS_PATH)
print("✅ Outputs:", [o.name for o in model.outputs])



# FASTAPI

# Create FastAPI app
app = FastAPI(title="SkinMeleo API", version="1.0.7")

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Base64 request body schema
class Base64Body(BaseModel):
    image_base64: str



# INFERENCE CORE

def run_inference_core(x: np.ndarray, debug: int = 0):
    # Run model prediction
    preds = model.predict(x, verbose=0)

    # Handle multi-head or dual-head models
    if isinstance(preds, (list, tuple)) and len(preds) == 3:
        mole_pred, binary_pred, type_pred = preds
        mole_score = float(mole_pred[0][0])
        image_suit = mole_score > MOLE_THRESHOLD

    elif isinstance(preds, (list, tuple)) and len(preds) == 2:
        binary_pred, type_pred = preds
        mole_score = 1.0
        image_suit = True

    else:
        raise RuntimeError(f"Unexpected model outputs: {type(preds)} / len={len(preds) if isinstance(preds,(list,tuple)) else 'N/A'}")

    # Extract scores
    bin_score = float(binary_pred[0][0])
    type_scores = type_pred[0]
    pred_type = CLASS_NAMES[int(np.argmax(type_scores))]

    # Final abnormal decision logic
    malignant_types = {"MEL", "BCC", "AKIEC"}
    is_abnormal = (bin_score > ABNORMAL_THRESHOLD) or (pred_type in malignant_types)

    # Debug response
    if debug:
        return {
            "Image_Suitability": bool(image_suit),
            "IsAbnormal": bool(is_abnormal) if image_suit else False,
            "Predicted_Type": pred_type if image_suit else "",
            "Scores": {
                "mole": mole_score,
                "binary": bin_score,
                "types": {CLASS_NAMES[i]: float(type_scores[i]) for i in range(len(CLASS_NAMES))}
            }
        }

    # Standard response
    if not image_suit:
        return {
            "Image_Suitability": False,
            "IsAbnormal": False,
            "Predicted_Type": ""
        }

    return {
        "Image_Suitability": True,
        "IsAbnormal": bool(is_abnormal),
        "Predicted_Type": pred_type
    }



# ROUTES

# API health check
@app.get("/health")
def health():
    return {"status": "ok"}


# ✅ UPDATED: Image prediction endpoint (accepts BOTH UploadFile and Base64 JSON)
@app.post("/predict")
async def predict(
    request: Request,
    image: UploadFile | None = File(default=None),
    authorization: str | None = Header(default=None),
    debug: int = Query(0)
):
    authorize(authorization)

    data: bytes | None = None

    # Case 1: Proper multipart upload (expected)
    if image is not None:
        data = await image.read()
    else:
        # Case 2: FlutterFlow sometimes sends JSON/base64 or wrong content-type.
        # Try reading JSON body with "image_base64"
        try:
            body = await request.json()
        except Exception:
            body = None

        if isinstance(body, dict) and "image_base64" in body and body["image_base64"]:
            try:
                data = base64.b64decode(body["image_base64"], validate=True)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid image_base64")
        else:
            # If no file and no base64, return a clear error instead of silent 422
            raise HTTPException(
                status_code=400,
                detail="No image provided. Send multipart/form-data with field 'image' (file) OR JSON with 'image_base64'."
            )

    x = preprocess_rgb(data)
    return JSONResponse(run_inference_core(x, debug=debug))


# Base64 image prediction endpoint (unchanged)
@app.post("/predict_base64")
def predict_base64(
    body: Base64Body,
    authorization: str | None = Header(default=None),
    debug: int = Query(0)
):
    authorize(authorization)

    try:
        raw = base64.b64decode(body.image_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image_base64")

    x = preprocess_rgb(raw)
    return JSONResponse(run_inference_core(x, debug=debug))
