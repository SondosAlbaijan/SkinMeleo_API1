FROM python:3.11-slim

# إعدادات مفيدة
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# حزم نظام خفيفة يحتاجها Pillow/TF (+ build-essential احتياط)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    curl git \
 && rm -rf /var/lib/apt/lists/*

# تثبيت المتطلبات
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# نسخ بقية الملفات (تأكد models/ موجود وما هو مستثنى)
COPY . /app

# المنصة ستمرر $PORT؛ لو ما مررته نستخدم 8000
EXPOSE 8000

# لا تثبّت ENV PORT ثابت
# ENV PORT=8080  <-- اشطبيه

# شغّل Uvicorn ببورت ديناميكي
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
