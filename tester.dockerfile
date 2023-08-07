# load the base image
FROM --platform=arm64 python:3.11-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# set working directory
WORKDIR /app

# copy reqs
COPY requirements.txt requirements.txt
COPY requirements_tests.txt requirements_tests.txt

# install reqs
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_tests.txt --no-cache-dir

# copy folders with data
COPY data data

# copy model path
COPY docker_model docker_model

# copy pyproject.toml
COPY pyproject.toml pyproject.toml

# copy source code

COPY src/data src/data
COPY src/model src/model

# run the test, make sure the model is named "best-checkpoint.ckpt"
CMD ["python", "-m", "src.model.eval", "data/raw", "docker_model/"]
