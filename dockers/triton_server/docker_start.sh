docker run -idt --gpus=all --restart=always --privileged=true -p8000:8000 -p8001:8001 -p8002:8002 -v /mnt/c/Users/fm/Desktop/infer_service/models:/model_repo dl_infer_server tritonserver --model-repository /model_repo/