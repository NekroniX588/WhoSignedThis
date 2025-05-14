# Use Python 3.8 as base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p uploads

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"] 