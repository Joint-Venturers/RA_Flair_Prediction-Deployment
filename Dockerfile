# Dockerfile for Enhanced RA Flare Prediction System

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements_enhanced.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_enhanced.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p data models reports

# Run training pipeline to generate models (optional - can be done externally)
# RUN python enhanced_train_ra_model.py

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "enhanced_server:app", "--host", "0.0.0.0", "--port", "8000"]