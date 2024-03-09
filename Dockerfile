FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
# apt-get update や apt-get upgrade の前にNO_PUBLIC_KEY対策
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt update
RUN apt -y upgrade
RUN apt -y install python3.8
RUN apt update && apt install -y python3-pip && apt install -y git

RUN pip3 install --upgrade pip
RUN pip install --upgrade pip
RUN apt install libjpeg-dev -y
RUN apt install vim -y
RUN apt install dstat -y

RUN pip install pytorch-lightning
RUN pip install transformers
RUN pip install datasets
RUN pip install roboflow
RUN pip install tqdm
RUN pip install torchvision
RUN pip install scikit-learn
RUN pip install torchmetrics
RUN pip install tqdm
# For Tensorboard
## GUIに関するライブラリ
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y python3-tk
RUN pip install ipywidgets
## UI
RUN pip install tensorflow --ignore-installed --user
RUN pip install tensorboardx

RUN echo 'export QT_X11_NO_MITSHM=1' >> ~/.bashrc
# RUN source ~/.bashrc

# イメージのエクスポート対策でコメントアウト
# RUN echo 'export DISPLAY=":0.0"' >> ~/.bashrc

WORKDIR /home/user
COPY segformer.py  /home/user
COPY segformer_inference.py  /home/user
COPY encode_decode_test.py  /home/user
COPY train_masks.csv  /home/user
COPY split_data/  /home/user/split_data
COPY split_data_small/  /home/user/split_data_small
COPY checkpoint/  /home/user/checkpoint