FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# pointing Cahce into /app/.cache to avoid permission issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.cache
ENV HF_HOME=/app/.cache

# create a non-root user and group for running the application safely (as "algorithm")
RUN groupadd -r algorithm && useradd -r -g algorithm algorithm

# give "algorithm" user ownership of the /app, /input, /output, and /app/.cache working directories
RUN mkdir -p /app /input /output /app/.cache \
    && chown -R algorithm:algorithm /app /input /output /app/.cache

WORKDIR /app

# install dependencies from requirements.txt as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the application code and model files into the container, changing ownership to "algorithm"
COPY --chown=algorithm:algorithm . /app

# switch from root to the non-root user "algorithm" for running the application
USER algorithm

# run the inference script when the container starts
CMD ["python", "inference.py"]
