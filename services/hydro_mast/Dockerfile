FROM python:3.11-slim

WORKDIR /app

# Runtime dependencies for numpy/pandas/torch CPU wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY . .

EXPOSE 8787

CMD ["python", "03_realtime_pipeline/realtime_dashboard_server.py", "--host", "0.0.0.0", "--port", "8787", "--no-open"]
