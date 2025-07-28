FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python packages with specific PyTorch CPU index
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Download NLTK data (with error handling)
RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)" || echo "NLTK download skipped - will download at runtime"

# Set environment variables to ensure CPU-only execution
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_CPP_MIN_LOG_LEVEL=2

COPY . .

# The model will be downloaded at runtime when first needed
CMD ["python", "1b.py", "--input_json", "./input_config.json", "--pdf_folder", "./pdf_folder", "--output_folder", "./1b_output", "--top_k", "5", "--use_mmr"]
