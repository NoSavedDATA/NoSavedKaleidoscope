FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        lsb-release \
        wget \
        curl \
        make \
        software-properties-common \
        gnupg && \
    rm -rf /var/lib/apt/lists/*

# Copy llvm.sh into the image (assumes llvm.sh is in the build context)
COPY llvm.sh /root/llvm.sh
RUN chmod +x /root/llvm.sh && /root/llvm.sh 19 all

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        llvm \
        clang \
        zlib1g-dev \
        libzstd-dev \
        libeigen3-dev \
        libopencv-dev && \
    rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]

