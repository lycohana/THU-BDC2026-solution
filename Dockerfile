FROM python:3.12-slim-bookworm

# Install build dependencies. This layer is intentionally kept identical to the
# previously successful build so Docker can reuse the local cache.
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Install ta-lib C library.
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make -j1 && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Do not run uv sync: uv.lock pins CUDA PyTorch dependencies. Install only the
# runtime packages used by train.py/test.py, with CPU PyTorch.
RUN uv venv .venv && \
    uv pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match \
        torch==2.6.0+cpu && \
    uv pip install \
        joblib==1.5.2 \
        lightgbm==4.6.0 \
        numpy==2.3.3 \
        pandas==2.3.2 \
        scikit-learn==1.7.2 \
        scipy==1.16.2 \
        ta-lib==0.6.8 \
        tensorboardX==2.6.4 \
        tqdm==4.67.1 && \
    .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

COPY . .

RUN if [ -d app/model ] && [ ! -d model ]; then cp -a app/model ./model; fi && \
    mkdir -p /app/data /app/model /app/output /app/temp

ENV PATH="/app/.venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib"

CMD ["sleep", "infinity"]
