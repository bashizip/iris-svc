FROM python:3.11-slim

# Create non-root user
RUN useradd -m appuser
WORKDIR /app

# System deps (optional but often useful)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Train and embed the model at build time
COPY train.py .
RUN python train.py

# App code
COPY app.py .

# Drop privileges
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
