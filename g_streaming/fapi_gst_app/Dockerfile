FROM ubuntu:22.04 AS opencv-builder

RUN apt update && \
    apt install -y --no-install-recommends \
        git \
        cmake \
        build-essential \
        curl \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        liblapack-dev \
        libgtk2.0-dev

RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common gpg-agent && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip3 install \
        setuptools \
        numpy

RUN git clone https://github.com/opencv/opencv.git --depth 1 --branch 4.11.0

WORKDIR /opencv/build

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D INSTALL_C_EXAMPLES=OFF \
        -D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=ON \
        -D PYTHON3_EXECUTABLE=$(which python3) \
        -D PYTHON3_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])") \
        -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
        -D WITH_GSTREAMER=ON \
        -D BUILD_EXAMPLES=OFF .. && \
    make -j$(nproc)


FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        curl gpg-agent && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip3 install \
        setuptools \
        numpy

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        zlib1g-dev \
        build-essential \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-good \
        gstreamer1.0-libav \
        gstreamer1.0-plugins-ugly \
        gir1.2-gst-rtsp-server-1.0 \
        libgtk2.0-dev && \
    cp /usr/lib/python3/dist-packages/gi/_gi.cpython-310-x86_64-linux-gnu.so /usr/lib/python3/dist-packages/gi/_gi.cpython-312-x86_64-linux-gnu.so


COPY --from=opencv-builder /opencv /opencv

WORKDIR /opencv/build

RUN make install && \
    rm -rf /opencv

RUN pip3 install ultralytics && \
    pip3 uninstall opencv-python -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app


# COPY ./app /requirements.txt

# RUN pip3 install --upgrade -r /requirements.txt
