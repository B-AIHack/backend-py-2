# ----- base image -----------------
FROM python:3.11-slim

# ----- системные зависимости ------
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# ----- рабочая директория ----------
WORKDIR /app

# ----- Python deps ------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ----- приложение -------------------
COPY app/ .

# ----- запускаем --------------------
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]