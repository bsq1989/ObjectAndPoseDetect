# ObjectAndPoseDetect
this project is a project that implements end2end yolov7 object detection and rtmpose detection using nvidia triton server.
The code is arranged according to the model repository structure require by triton. All preprocess and postprocess are implement by opencv and also has a model config file.
Finally this repo also implement a hole endtoend procudure config which make preprocess , infer, postprocess all together. As yolo and rtmpose is not a very heavy load for GPU on server, only single batch inference is implement, but the input dim is reserved. You can also develop batch request on triton to reach a higher throughput
