FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY server/ .
COPY models.py .

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
