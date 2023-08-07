# base image
FROM --platform=arm64 python:3.11-slim

# install python

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# setup working directory
WORKDIR /app

# copy reqs
COPY requirements.txt requirements.txt
COPY requirements_tests.txt requirements_tests.txt

# install requirements
# --no-cache-dir to keep image as small as possible
RUN pip3 install -r requirements.txt --no-cache-dir
RUN pip3 install -r requirements_tests.txt --no-cache-dir

# copy folders with data
COPY data data

# create model output folder
RUN mkdir -p model

# copy pyproject.toml
COPY pyproject.toml pyproject.toml

# copy source code
COPY src/data src/data
COPY src/model src/model

# the docker will first train a model and then test the model
CMD ["python", "-m", "src.model.train", "data/raw", "model"]
