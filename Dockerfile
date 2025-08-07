# 1. Base Image  
# Using a PyTorch image that supports CUDA 12.1 and Python 3.10.x
# Example: pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime (Python 3.10.12)
# Check Docker Hub for the latest appropriate pytorch/pytorch image compatible with Python 3.10 and CUDA 12.1
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# 2. Environment Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_HTTP_TIMEOUT=3600
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PATH="/usr/local/bin:${PATH}"


# 3. System Packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    curl \
    xz-utils \
    build-essential \
    python3-dev \
    cmake \
    libglib2.0-0 \
    git \
    wget \
    # other system dependencies from cog.yaml if any were missed (libgl1 is listed)
 && rm -rf /var/lib/apt/lists/*

# Install static ffmpeg
RUN curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o ffmpeg-release-amd64-static.tar.xz && \
    mkdir -p /opt/ffmpeg && \
    tar -xf ffmpeg-release-amd64-static.tar.xz -C /opt/ffmpeg --strip-components=1 && \
    rm ffmpeg-release-amd64-static.tar.xz && \
    ln -s /opt/ffmpeg/ffmpeg /usr/local/bin/ffmpeg && \
    ln -s /opt/ffmpeg/ffprobe /usr/local/bin/ffprobe

# 4. Install pget
RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.10.2/pget_linux_x86_64" \
 && chmod +x /usr/local/bin/pget

# 5. Set up Working Directory
WORKDIR /app

# 6. Copy Requirements
COPY requirements.txt .

# 7. Install Python Dependencies
# The --extra-index-url is part of the requirements.txt content itself, so pip should pick it up.
# If not, it needs to be specified here. Assuming it's handled if present in the file.
RUN pip install --no-cache-dir -r requirements.txt hf_transfer
RUN pip install huggingface_hub --quiet

# 创建MODNet模型目录（模型将在运行时下载）
RUN mkdir -p /app/checkpoints/modnet

# 下载MODNet源码和模型
RUN git clone https://github.com/ZHKKKe/MODNet.git /tmp/modnet && \
    cp -r /tmp/modnet/src /app/modnet_src && \
    rm -rf /tmp/modnet
    
# 下载MODNet模型（人像抠图专用）
RUN pget https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/models/modnet_photographic_portrait_matting.ckpt /app/checkpoints/modnet/modnet_photographic_portrait_matting.ckpt || \
    curl -L https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/models/modnet_photographic_portrait_matting.ckpt -o /app/checkpoints/modnet/modnet_photographic_portrait_matting.ckpt

# 8. Copy Project Files
COPY . .

# 9. Download checkpoints
RUN huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints
RUN huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints

# 10. Expose Port (Good practice, RunPod might manage ports differently)
EXPOSE 8080

# 10. Command to start the RunPod worker
# This assumes 'runpod_handler.py' contains a function 'handler'
# and that the 'runpod' Python package provides the serverless entry point.
# The user should verify this CMD with RunPod's documentation.
CMD ["python", "runpod_handler.py"]