# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install system dependencies for matplotlib, seaborn, imbalanced-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libpng-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose the port Flask will run on
EXPOSE 8000

# Run the Flask app
CMD ["python", "main_flask.py"]
