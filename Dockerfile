# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

ARG PYTHON_VERSION=3.12.2
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/bin/bash" \
    --uid "${UID}" \
    appuser

# Create necessary directories and set permissions
RUN mkdir -p /home/appuser/.cache /home/appuser/.config && \
    chown -R appuser:appuser /home/appuser && \
    chown -R appuser:appuser /app

# Set environment variables for cache and config directories
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface
ENV MPLCONFIGDIR=/home/appuser/.config/matplotlib

COPY requirements.txt .

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN python -m pip install --no-cache-dir -r requirements.txt

# Switch to the non-privileged user to run the application.
USER appuser

# Copy the source code into the container
COPY --chown=appuser:appuser . .

# Expose the port that the application listens on.
EXPOSE 8080

# Run the application.
CMD python -m hypercorn api_server:app --bind=0.0.0.0:8080
