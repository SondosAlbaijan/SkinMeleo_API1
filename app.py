# app.py
import os, io, zipfile, tempfile, base64
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras import Model

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ================== إعدادات عامة ==================
IMG_SIZE = (240, 240)

# نفس ترتيب ISIC
CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# عتبة وجود الشامة (تقدرين تعدلينها لاحقاً)
MOLE_THRESHOLD = 0.80   # لو أقل من 0.8 نعتبر الصورة غير مناسبة للتحليل

# عتبة اعتباره "Abnormal"
ABNORMAL_THRESHOLD = 0.60   # لو أكبر من 0.6 نعتبرها خبيثة

# مفتاح الـ API (خليه فاضي = بدون حماية)
API_KEY = ""

# مسارات الملفات
WEIGHTS_PATH   = os.getenv("WEIGHTS_PATH", "models/model.weights.h5")
KERAS_PRIMARY  = os.getenv("KERAS_PRIMARY", "models/api_export.keras")
KERAS_FALLBACK = os.getenv("KERAS_FALLBACK", "models/SkinMeleo_model.keras")


# ================== أدوات الموديل ==================
def build_model():
    """
    يبني نفس معمارية الموديل اللي دربتوه:
    EfficientNetB1 + GlobalAveragePooling2D + 3 مخارج.
    نستخدم weights=None لأننا بنحمل وزناتكم أنتم.
    """
    base = tf.keras.applications.EfficientNetB1(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights=None  # مهم: لا نحمّل ImageNet هنا
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
    """
    يرجّع مسار model.weights.h5
    لو ما لقاه، يحاول يستخرجه من ملف .keras (اللي حملتوه من كولاب).
    """
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
    """
    يحوّل الصورة إلى RGB بالحجم 240x240
    مع preprocess_input تبع EfficientNet.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, 0)  # (1, 240, 240, 3)


def authorize(auth_header: str | None):
    """
    حماية بسيطة بالمفتاح (Bearer <API_KEY>).
    لو API_KEY = "" ما يتطبق شيء.
    """
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
    print(f"✅ Loaded weights from: {weights_file}")
except Exception as e:
    print("⚠️ direct load_weights failed, trying by_name+skip_mismatch:", e)
    model.load_weights(weights_file, by_name=True, skip_mismatch=True)

# تمريرة فحص سريعة
_ = model(tf.zeros((1,) + IMG_SIZE + (3,)))
print("✅ Model ready. Input:", model.input_shape)


# ================== إعداد FastAPI ==================
app = FastAPI(title="SkinMeleo API", version="1.0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # تقدرين تضيقينها لاحقاً
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Base64Body(BaseModel):
    image_base64: str


@app.get("/health")
def health():
    return {"status": "ok"}


# ================== /predict (ملف مرفوع) ==================
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

    mole_score   = float(mole_pred[0][0])
    binary_score = float(binary_pred[0][0])
    type_vector  = type_pred[0]

    # هل الصورة مناسبة للتحليل (فيها شامة بنسبة كافية)؟
    image_suit = bool(mole_score >= MOLE_THRESHOLD)

    # أقرب نوع دائماً = argmax (زي ما طلبتي)
    top_idx  = int(np.argmax(type_vector))
    pred_type = CLASS_NAMES[top_idx]

    # IsAbnormal يعتمد على binary_score فقط لو الصورة مناسبة
    is_abnormal = bool(binary_score > ABNORMAL_THRESHOLD) if image_suit else False

    if debug:
        return {
            "IsAbnormal": is_abnormal if image_suit else False,
            "Image_Suitability": image_suit,
            "Predicted_Type": pred_type if image_suit else "",
            "scores": {
                "mole": mole_score,
                "binary": binary_score,
                "types": {CLASS_NAMES[i]: float(type_vector[i]) for i in range(7)}
            }
        }

    # لو الصورة غير مناسبة، نمشي حسب المطلوب: النوع فاضي
    if not image_suit:
        return {
            "IsAbnormal": False,
            "Image_Suitability": False,
            "Predicted_Type": ""
        }

    # صورة مناسبة → نرجّع أقرب نوع + الحالة
    return {
        "IsAbnormal": is_abnormal,
        "Image_Suitability": True,
        "Predicted_Type": pred_type
    }


# ================== /predict_base64 (صورة Base64) ==================
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

    mole_score   = float(mole_pred[0][0])
    binary_score = float(binary_pred[0][0])
    type_vector  = type_pred[0]

    image_suit = bool(mole_score >= MOLE_THRESHOLD)

    top_idx   = int(np.argmax(type_vector))
    pred_type = CLASS_NAMES[top_idx]

    is_abnormal = bool(binary_score > ABNORMAL_THRESHOLD) if image_suit else False

    if debug:
        return {
            "IsAbnormal": is_abnormal if image_suit else False,
            "Image_Suitability": image_suit,
            "Predicted_Type": pred_type if image_suit else "",
            "scores": {
                "mole": mole_score,
                "binary": binary_score,
                "types": {CLASS_NAMES[i]: float(type_vector[i]) for i in range(7)}
            }
        }

    if not image_suit:
        return {
            "IsAbnormal": False,
            "Image_Suitability": False,
            "Predicted_Type": ""
        }

    return {
        "IsAbnormal": is_abnormal,
        "Image_Suitability": True,
        "Predicted_Type": pred_type
    }
