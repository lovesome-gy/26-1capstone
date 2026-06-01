FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir psycopg2-binary requests python-dotenv
COPY app/ ./app/
CMD ["python", "-m", "app.main"]