FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY api ./api
COPY dashboard ./dashboard
COPY data/seed ./data/seed
COPY data/corpus ./data/corpus
RUN pip install --no-cache-dir -e ".[serve]"
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
