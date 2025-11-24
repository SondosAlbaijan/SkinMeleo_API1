import os, io, base64
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import tensorflow as tf
from tensorflow.keras.models import load_model

from PIL import Image, ImageOps

# محاولة تفعيل دعم AVIF لو البلَغن موجود
try:
    import pillow_avif  # noqa: F401
except Exception:
    pass


# ==============================
# CONFIG
# ==============================
IMG_SIZE = (240, 240)
CLASS_NAMES = ["MEL","NV","BCC","AKIEC","BKL","DF","VASC"]

MOLE_THRESHOLD = 0.99
ABNORMAL_THRESHOLD = 0.50

API_KEY = ""

KERAS_PRIMARY  = os.getenv("KERAS_PRIMARY", "models/api_export.keras")
KERAS_FALLBACK = os.getenv("KERAS_FALLBACK", "models/SkinMeleo_model.keras")


# ==============================
# PREPROCESS (مطابق كولاب 1:1)
# ==============================
def preprocess_rgb(file_bytes: bytes) -> np.ndarray:
    # نفس load_img في كولاب: فتح عبر PIL
    img = Image.open(io.BytesIO(file_bytes))

    # نفس كيراس: تصحيح اتجاه الصورة لو فيها EXIF
    img = ImageOps.exif_transpose(img).convert("RGB")

    # نفس default في keras load_img: resize بـ NEAREST
    img = img.resize(IMG_SIZE, resample=Image.NEAREST)

    # نفس img_to_array: float32 بدون /255
    arr = np.asarray(img, dtype=np.float32)

    # نفس preprocess_input حق EfficientNet
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)

    return np.expand_dims(arr, 0)


# ==============================
# AUTH
# ==============================
def authorize(auth_header: str | None):
    if not API_KEY:
        return
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing/invalid Authorization header")
    token = auth_header.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ==============================
# LOAD MODEL (نفس الملف نفسه بدون rebuild)
# ==============================
KERAS_PATH = KERAS_PRIMARY if os.path.exists(KERAS_PRIMARY) else KERAS_FALLBACK
if not os.path.exists(KERAS_PATH):
    raise RuntimeError("لم يتم العثور على ملف .keras داخل models/")

model = load_model(KERAS_PATH, compile=False)

# warmup
_ = model(tf.zeros((1,) + IMG_SIZE + (3,)))
print("✅ Loaded model from:", KERAS_PATH)
print("✅ Outputs:", [o.name for o in model.outputs])


# ==============================
# FASTAPI
# ==============================
app = FastAPI(title="SkinMeleo API", version="1.0.7")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class Base64Body(BaseModel):
    image_base64: str


# ==============================
# INFERENCE CORE
# ==============================
def run_inference_core(x: np.ndarray, debug: int = 0):
    preds = model.predict(x, verbose=0)

    # ---- دعم رأسين أو ثلاثة
    if isinstance(preds, (list, tuple)) and len(preds) == 3:
        mole_pred, binary_pred, type_pred = preds
        mole_score = float(mole_pred[0][0])
        image_suit = mole_score > MOLE_THRESHOLD
    elif isinstance(preds, (list, tuple)) and len(preds) == 2:
        binary_pred, type_pred = preds
        mole_score = 1.0
        image_suit = True  # ما عندك mole head أصلاً
    else:
        raise RuntimeError(f"Unexpected model outputs: {type(preds)} / len={len(preds) if isinstance(preds,(list,tuple)) else 'N/A'}")

    # ---- Scores
    bin_score = float(binary_pred[0][0])
    type_scores = type_pred[0]
    pred_type = CLASS_NAMES[int(np.argmax(type_scores))]

    # ---- منطق التماسك: إذا النوع خبيث اعتبر Abnormal حتى لو bin منخفض
    malignant_types = {"MEL", "BCC", "AKIEC"}
    is_abnormal = (bin_score > ABNORMAL_THRESHOLD) or (pred_type in malignant_types)

    # ---- Debug output
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

    # ---- Normal output
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




# ==============================
# ROUTES
# ==============================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    authorization: str | None = Header(default=None),
    debug: int = Query(0)
):
    authorize(authorization)
    data = await image.read()
    x = preprocess_rgb(data)
    return JSONResponse(run_inference_core(x, debug=debug))


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
        raise HTTPException(status_code=400, detail="image_base64 غير صالح")

    x = preprocess_rgb(raw)
    return JSONResponse(run_inference_core(x, debug=debug))
