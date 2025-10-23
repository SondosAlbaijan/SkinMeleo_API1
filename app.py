# app.py
import os, io, zipfile, tempfile, base64
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras import Model

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ================== إعدادات ==================
IMG_SIZE = (240, 240)
CLASS_NAMES = ["MEL","NV","BCC","AKIEC","BKL","DF","VASC"]

MOLE_THRESHOLD = 0.90          # ↓ خفّضناها من 0.99
ABNORMAL_THRESHOLD = 0.50

API_KEY = ""  # خليها "" لو ما تبين حماية

# مسارات الملفات
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "models/model.weights.h5")
KERAS_PRIMARY = os.getenv("KERAS_PRIMARY", "models/api_export.keras")
KERAS_FALLBACK = os.getenv("KERAS_FALLBACK", "models/SkinMeleo_model.keras")

# ================== أدوات ==================
def build_model():
    """نفس المعمارية بدون وزنات ImageNet؛ بنحمّل وزناتك فقط."""
    base = tf.keras.applications.EfficientNetB1(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights=None  # مهم: لا نحمّل ImageNet
    )
    base.trainable = False

    inp = Input(shape=IMG_SIZE + (3,))
    x = base(inp, training=False)
    x = GlobalAveragePooling2D()(x)
    mole   = Dense(1, activation="sigmoid", name="mole_presence_output")(x)
    binary = Dense(1, activation="sigmoid", name="binary_output")(x)
    ctype  = Dense(7, activation="softmax",  name="cancer_type_output")(x)
    model = Model(inp, [mole, binary, ctype])
    return model

def ensure_weights_file() -> str:
    """يرجع مسار H5. لو مو موجود، يحاول يستخرجه من ملف .keras (zip)."""
    if os.path.exists(WEIGHTS_PATH):
        return WEIGHTS_PATH

    for keras_path in (KERAS_PRIMARY, KERAS_FALLBACK):
        if os.path.exists(keras_path) and zipfile.is_zipfile(keras_path):
            with zipfile.ZipFile(keras_path, "r") as z:
                cand = [n for n in z.namelist() if n.endswith("model.weights.h5")]
                if cand:
                    tmpdir = tempfile.mkdtemp(prefix="weights_")
                    out = os.path.join(tmpdir, "model.weights.h5")
                    with z.open(cand[0]) as src, open(out, "wb") as dst:
                        dst.write(src.read())
                    print(f"✅ extracted weights from {keras_path}")
                    return out
    raise RuntimeError(
        "لا يوجد models/model.weights.h5 ولا قدرت أستخرجه من أي .keras داخل models/."
    )

def preprocess_rgb(file_bytes: bytes) -> np.ndarray:
    """RGB 240x240 مع preprocess_input تبع EfficientNet."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, 0)  # (1,240,240,3)

def authorize(auth_header: str | None):
    """حماية بسيطة بالمفتاح. عطّليها بخلي API_KEY = ''."""
    if not API_KEY:
        return
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing/invalid Authorization header")
    token = auth_header.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ================== تحميل الموديل مرة واحدة ==================
weights_file = ensure_weights_file()
model = build_model()
try:
    model.load_weights(weights_file)
except Exception:
    # لو اختلاف أسماء بسيط
    model.load_weights(weights_file, by_name=True, skip_mismatch=True)

# جولة تمرين/فحص سريعة للتأكد
_ = model(tf.zeros((1,) + IMG_SIZE + (3,)))
print("✅ Model ready. Input:", model.input_shape)

# ================== FastAPI ==================
app = FastAPI(title="SkinMeleo API", version="1.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class Base64Body(BaseModel):
    image_base64: str

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
    mole_pred, binary_pred, type_pred = model.predict(x, verbose=0)

    image_suit = bool(mole_pred[0][0] >= MOLE_THRESHOLD)
    is_abnormal = bool(binary_pred[0][0] > ABNORMAL_THRESHOLD)
    pred_type = CLASS_NAMES[int(np.argmax(type_pred[0]))]

    if debug:
        return {
            "IsAbnormal": is_abnormal if image_suit else False,
            "Image_Suitability": image_suit,
            "Predicted_Type": pred_type if image_suit else "",
            "scores": {
                "mole": float(mole_pred[0][0]),
                "binary": float(binary_pred[0][0]),
                "types": {CLASS_NAMES[i]: float(type_pred[0][i]) for i in range(7)}
            }
        }

    if not image_suit:
        return {"IsAbnormal": False, "Image_Suitability": False, "Predicted_Type": ""}

    return {"IsAbnormal": is_abnormal, "Image_Suitability": True, "Predicted_Type": pred_type}

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
    mole_pred, binary_pred, type_pred = model.predict(x, verbose=0)

    image_suit = bool(mole_pred[0][0] >= MOLE_THRESHOLD)
    is_abnormal = bool(binary_pred[0][0] > ABNORMAL_THRESHOLD)
    pred_type = CLASS_NAMES[int(np.argmax(type_pred[0]))]

    if debug:
        return {
            "IsAbnormal": is_abnormal if image_suit else False,
            "Image_Suitability": image_suit,
            "Predicted_Type": pred_type if image_suit else "",
            "scores": {
                "mole": float(mole_pred[0][0]),
                "binary": float(binary_pred[0][0]),
                "types": {CLASS_NAMES[i]: float(type_pred[0][i]) for i in range(7)}
            }
        }

    if not image_suit:
        return {"IsAbnormal": False, "Image_Suitability": False, "Predicted_Type": ""}

    return {"IsAbnormal": is_abnormal, "Image_Suitability": True, "Predicted_Type": pred_type}
