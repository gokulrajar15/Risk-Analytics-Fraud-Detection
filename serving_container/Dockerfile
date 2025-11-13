FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY serving_container/requirements.txt .

RUN uv pip install -r requirements.txt --system

COPY serving_container/ ./serving_container/
COPY .env .

CMD ["python", "serving_container/app.py"]