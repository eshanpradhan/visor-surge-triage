# VISOR — Hugging Face Space (Docker SDK)
#
# Docker rather than the Streamlit SDK so the Python version and the wheel index
# are pinned here instead of being auto-detected. Auto-detection is what broke the
# earlier deploy: the platform chose Python 3.14, where several dependencies have
# no wheels, and pip fell back to compiling from source.

FROM python:3.11-slim

# libgomp1 is LightGBM's OpenMP runtime. Its wheels are py3-none-manylinux and do
# not vendor it, so importing lightgbm on a slim image fails without this.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as UID 1000. Creating that user here means
# HOME, the pip install target and the torch cache are all writable at runtime;
# running as root instead leaves a read-only HOME and the weight cache fails.
RUN useradd -m -u 1000 user
USER user

# VISOR_DEMO_MODE: the Space has no SBU cohort files, so app.py would fall back to
# demo mode on its own. Set explicitly so the deployed behaviour is declared rather
# than inferred from a missing path.
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/home/user/.cache/torch \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    VISOR_DEMO_MODE=1

WORKDIR $HOME/app

# Dependencies first, so a code change does not reinstall torch.
COPY --chown=user requirements.txt ./

# torch and torchvision come from the CPU wheel index. The default PyPI wheels
# bundle CUDA and are several GB; a Space has no GPU, so that is pure image size
# and pull time. Versions match requirements.txt exactly -- "==2.12.0" is
# satisfied by "2.12.0+cpu" under PEP 440 -- so the numerics are unchanged from
# what was validated locally.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.12.0 torchvision==0.27.0 \
    && pip install --no-cache-dir -r requirements.txt

# Bake the frozen ImageNet backbone into the image. The checkpoint in models/
# holds only the 68 trained tensors; torchvision fetches the rest (~98 MB) on
# first use. Downloading it at build time means a cold start does not wait on
# download.pytorch.org, which matters because a free Space re-downloads on every
# wake, not just the first boot ever.
RUN python -c "import torch; from torchvision.models import ResNet50_Weights; \
    torch.hub.load_state_dict_from_url(ResNet50_Weights.IMAGENET1K_V1.url, progress=False)"

COPY --chown=user . .

# Fail the build rather than the request if an artifact is missing or the demo
# cases cannot be scored. A broken Space that starts is worse than one that does
# not build.
RUN python -c "import demo_data; frame, encoder = demo_data.build_demo_frame(); \
    matrix = encoder.transform(frame); \
    assert matrix.shape[0] == len(frame), 'demo frame failed to encode'; \
    print(f'demo cases OK: {matrix.shape}')"

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
