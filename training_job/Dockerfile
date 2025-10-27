FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY training_job/requirements.txt .

RUN uv pip install -r requirements.txt --system

# Copy the training_job directory contents into the container

COPY training_job/ ./training_job/
COPY .env .

ENTRYPOINT ["python", "training_job/training/train.py"]