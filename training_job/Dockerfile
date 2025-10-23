FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-7:latest
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "train.py"]
