trtexec --onnx=yolov7.onnx \
--minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640 \
--fp16 --workspace=4096 \
--saveEngine=yolov7-fp16-1x8x8.engine --timingCacheFile=timing.cache


#docker run -ti --rm --gpus=all --network=host -v $PWD:/mnt --name triton-client nvcr.io/nvidia/tritonserver:23.09-py3-sdk
