# Use cuda.Dockerfile if you have a CUDA-enabled GPU
FROM python:3.11-slim-buster
WORKDIR /app
ADD . /app
RUN apt-get update && apt-get install -y \
    libhdf5-serial-dev \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    h5py \
    matplotlib \
    numpy \
    pandas \
    tables \
    scikit-learn \
    scipy \
    tqdm \
    spatialprofilingtoolbox[cggnn]
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir dgl -f https://data.dgl.ai/wheels/repo.html
RUN pip install --no-cache-dir dglgo -f https://data.dgl.ai/wheels-test/repo.html
ENV DGLBACKEND=pytorch

EXPOSE 80

ENTRYPOINT ["python", "main.py"]
CMD ["--cg_directory", ".", "-b", "1", "--epochs", "10", "-l", "0.001", "-k", "0"]
