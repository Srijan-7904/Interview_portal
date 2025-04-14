# Use Python 3.8 base image (matches Render's default)
FROM python:3.8-slim

# Install system dependencies for dlib, OpenCV, Java, and Node.js
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    openjdk-17-jdk \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (for JavaScript code execution)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data for textblob
RUN python -m nltk.downloader punkt averaged_perceptron_tagger

# Copy shape_predictor_68_face_landmarks.dat
RUN mkdir -p models
COPY models/shape_predictor_68_face_landmarks.dat models/

# Copy application code
COPY . .

# Expose port (Render assigns dynamically, but include for clarity)
EXPOSE 5000

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]