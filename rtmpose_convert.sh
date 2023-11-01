trtexec --onnx=end2end.onnx \
--minShapes=input:1x3x384x288 --optShapes=input:8x3x384x288 --maxShapes=input:8x3x384x288 \
--fp16 --workspace=4096 --saveEngine=rtmpose-fp16-1x8x8.engine --timingCacheFile=timing.cache