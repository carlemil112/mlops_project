FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3.11-distutils curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip /usr/bin/pip

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip setuptools && pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "pytest", "-q"]