FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY predict.py .
COPY model_growth.pkl .

# Cloud Run uses PORT env variable
ENV PORT=8080

EXPOSE 8080

# Use gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 1 --threads 2 --timeout 60 app:app
