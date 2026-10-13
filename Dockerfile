FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
# torch already ships in the base image
RUN grep -v '^torch' requirements.txt > deps.txt && pip install --no-cache-dir -r deps.txt

COPY configs configs
COPY echoseg echoseg
COPY scripts scripts
COPY tests tests
COPY pyproject.toml .

# docker run --gpus all -v $PWD/data:/app/data -v $PWD/runs:/app/runs echoseg-rt echoseg.train
ENTRYPOINT ["python", "-m"]
CMD ["echoseg.benchmark"]
