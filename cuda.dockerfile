FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
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
RUN pip install --no-cache-dir dgl -f https://data.dgl.ai/wheels/cu118/repo.html
RUN pip install --no-cache-dir dglgo -f https://data.dgl.ai/wheels-test/repo.html

EXPOSE 80

ENTRYPOINT ["python", "main.py"]
CMD ["--cg_directory", ".", "-b", "1", "--epochs", "10", "-l", "0.001", "-k", "0"]
