FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./app .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

# CMD ["watchmedo", "auto-restart", "--directory=.", "--pattern=*.py", "--recursive", "python", "app.py"]
CMD ["python", "app.py"]