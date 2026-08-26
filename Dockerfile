# 1) ابدأ من نسخة بايثون رسمية وخفيفة
FROM python:3.11-slim

# 2) حدد مجلد العمل جوه الـ container
WORKDIR /app

# 3) انسخ ملف المتطلبات وثبّت المكتبات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) انسخ باقي ملفات المشروع
COPY . .

# 5) حدد الأمر اللي هيتنفذ لما الـ container يشتغل
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]