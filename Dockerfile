FROM python:3.9-slim
WORKDIR /app/

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY api.py get_mnist.py num_seq_generator.py ./
ENTRYPOINT ["python", "api.py"]