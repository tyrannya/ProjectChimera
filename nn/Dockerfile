FROM bentoml/model-server:python3.11-torch2.3
WORKDIR /app
COPY infer_service.py ./
RUN pip install --no-cache-dir bentoml torch
CMD ["bentoml", "serve", "infer_service:svc", "--port", "5000"]
