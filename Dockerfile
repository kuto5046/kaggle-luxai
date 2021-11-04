# pytorch versionに注意
# FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu20.04
# FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu18.04
FROM gcr.io/kaggle-gpu-images/python:v100

# 時間設定
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# ENV DEBIAN_FRONTEND=noninteractive
# install basic dependencies
RUN apt-get -y update && apt-get install -y \
    # sudo \
    # wget \
    # cmake \
    # vim \
    # git \
    # tmux \
    # zip \
    # unzip \
    # gcc \
    # g++ \
    # build-essential \
    # ca-certificates \
    # software-properties-common \
    # libsm6 \
    # libxext6 \
    # libxrender-dev \
    # libpng-dev \
    # libfreetype6-dev \
    # libgl1-mesa-dev \
    # libsndfile1 \
    # curl \
    zsh \
    xonsh \
    neovim
    # nodejs \
    # npm 


# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get update && \
#     apt-get install -y \
#     python3.7  \
#     python3.7-venv \
#     python3.7-dev \
#     python3-pip \
#     python3-ipdb

# node js を最新Verにする
# RUN npm -y install n -g && \
#     n stable && \
#     apt purge -y nodejs npm

# set path
# ENV PATH /usr/bin:$PATH

# install common python packages
# COPY ./requirements.txt /
# RUN pip3 install --upgrade pip && \
#     pip3 install -r /requirements.txt

# latest imitation library
RUN git clone http://github.com/HumanCompatibleAI/imitation && \
    cd imitation && \
    pip3 install -e .

# https://qiita.com/Hiroaki-K4/items/c1be8adba18b9f0b4cef
# RUN pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.htmlu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# add user
# ARG DOCKER_UID=1111
# ARG DOCKER_USER=user
# ARG DOCKER_PASSWORD=kuzira
# RUN useradd -m --uid ${DOCKER_UID} --groups sudo ${DOCKER_USER} \
#   && echo ${DOCKER_USER}:${DOCKER_PASSWORD} | chpasswd

# for user
RUN mkdir /root/.kaggle
COPY ./kaggle.json /root/.kaggle/
# set working directory
RUN mkdir /work
WORKDIR /work
# 本当はよくないがkaggle cliがuserで使えないので600 -> 666
RUN chmod 600 /root/.kaggle/kaggle.json

# switch user
# USER ${DOCKER_USER}

RUN git clone https://github.com/kuto5046/dotfiles.git /dotfiles
RUN bash /dotfiles/.bin/install.sh

# jupyter lab
# ENV PATH /home/user/.local/bin:$PATH
# RUN wget -q -O - https://linux.kite.com/dls/linux/current
