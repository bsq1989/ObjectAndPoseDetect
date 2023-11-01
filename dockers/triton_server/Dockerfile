FROM nvcr.io/nvidia/tritonserver:23.09-py3
RUN pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN apt update
RUN apt install libgl1-mesa-glx --assume-yes
RUN pip install --upgrade Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple