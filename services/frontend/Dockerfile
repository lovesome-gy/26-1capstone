FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir streamlit httpx plotly pandas
COPY app/ ./app/
EXPOSE 8501
CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]