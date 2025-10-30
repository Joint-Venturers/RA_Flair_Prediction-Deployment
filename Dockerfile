# Use Python 3.11 with Alpine Linux
FROM python:3.11-alpine

# Install system dependencies for building Python packages
RUN apk add --no-cache \
    gcc \
    musl-dev \
    linux-headers \
    g++ \
    gfortran \
    openblas-dev \
    lapack-dev

# Create working directory
WORKDIR /code

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY app/ ./app/

# Expose the port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/code

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
