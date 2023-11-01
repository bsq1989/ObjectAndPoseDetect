FROM nvcr.io/nvidia/tritonserver:23.09-py3-sdk
ENV TZ=Asia/Shanghai
ENV LANG en_US.UTF-8
RUN pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN apt install libgl1-mesa-glx --assume-yes
RUN apt-get install redis --assume-yes
RUN python -m pip install -U scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install filterpy -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install joblib -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install redis -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install oss2 -i https://pypi.tuna.tsinghua.edu.cn/simple