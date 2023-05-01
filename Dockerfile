FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime as pytorch-base

RUN apt-get update && apt-get install -y git wget unzip

RUN pip install colink flbenchmark scikit-learn

WORKDIR /
RUN git clone https://github.com/facebookresearch/CrypTen.git crypten_src
WORKDIR /crypten_src
COPY crypten.patch /crypten_src
RUN git checkout 891fa4709e4849ff50ace6933f20375b04b3f722 && \
    git apply crypten.patch
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN pip install .

WORKDIR /test
COPY src /test/src/
COPY setup.py /test/
RUN pip install .

VOLUME /data

ENTRYPOINT [ "unifed-crypten" ]
