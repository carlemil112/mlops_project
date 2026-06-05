FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Copenhagen

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-distutils curl \
    cmake g++ \
    libopencv-dev \
    wget unzip flatbuffers-compiler \
    git \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/local/bin/pip /usr/bin/pip
 
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip setuptools && pip install -r requirements.txt
 
# Download and build TFLite from source
RUN wget -q https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.13.0.zip && \
    unzip -q v2.13.0.zip && \
    rm v2.13.0.zip && \
    cmake -B tensorflow-2.13.0/tflite_build \
        tensorflow-2.13.0/tensorflow/lite \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build tensorflow-2.13.0/tflite_build -j$(nproc)
 
COPY . .
 
# Compile inference.cpp against the TFLite we just built
RUN g++ -O2 -std=c++17 -o inference inference.cpp \
    -I /usr/local/lib/python3.11/dist-packages/tensorflow/include \
    -L /usr/local/lib/python3.11/dist-packages/tensorflow \
    -l:libtensorflow_lite_c.so \
    $(pkg-config --cflags --libs opencv4) \
    -Wl,-rpath,/usr/local/lib/python3.11/dist-packages/tensorflow
 
CMD ["python", "-m", "pytest", "-q"]
 