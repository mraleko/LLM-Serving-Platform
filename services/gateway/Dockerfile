FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/libs/common

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY libs /app/libs
COPY services /app/services

EXPOSE 8000

CMD ["uvicorn", "services.gateway.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
