import grpc
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from processing import preprocess, postprocess
from render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS
from labels import COCOLabels
import numpy as np
import cv2
from tritonclient.grpc import service_pb2, service_pb2_grpc
import yaml


def test_client():
    triton_client = grpcclient.InferenceServerClient(
            url='localhost:8001')
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('images', [1, 3, 640, 640], "FP32"))
    outputs.append(grpcclient.InferRequestedOutput('num_dets'))
    outputs.append(grpcclient.InferRequestedOutput('det_boxes'))
    outputs.append(grpcclient.InferRequestedOutput('det_scores'))
    outputs.append(grpcclient.InferRequestedOutput('det_classes'))

    input_image = cv2.imread('./data/dog.jpg') 
    input_image_buffer = preprocess(input_image, [640,640])
    input_image_buffer = np.expand_dims(input_image_buffer, axis=0)   
    inputs[0].set_data_from_numpy(input_image_buffer)
    print("Invoking inference...")
    model = 'yolo_trt'
    results = triton_client.infer(model_name=model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=10.0)   
    statistics = triton_client.get_inference_statistics(model_name=model)
    print(statistics)
    num_dets = results.as_numpy('num_dets')
    det_boxes = results.as_numpy('det_boxes')
    det_scores = results.as_numpy('det_scores')
    det_classes = results.as_numpy('det_classes')   
    detected_objects = postprocess(num_dets, det_boxes, det_scores, det_classes, input_image.shape[1], input_image.shape[0], [640, 640])
    #print(f"Detected objects: {len(detected_objects)}")
    for box in detected_objects:
        print(f"{COCOLabels(box.classID).name}: {box.confidence}")
        input_image = render_box(input_image, box.box(), color=tuple(RAND_COLORS[box.classID % 64].tolist()))
        size = get_text_size(input_image, f"{COCOLabels(box.classID).name}: {box.confidence:.2f}", normalised_scaling=0.6)
        input_image = render_filled_box(input_image, (box.x1 - 3, box.y1 - 3, box.x1 + size[0], box.y1 + size[1]), color=(220, 220, 220))
        input_image = render_text(input_image, f"{COCOLabels(box.classID).name}: {box.confidence:.2f}", (box.x1, box.y1), color=(30, 30, 30), normalised_scaling=0.5)

    cv2.imwrite('./data/test_triton.png', input_image)
    print(f"Saved result to")




def test_preprocess_client():
    triton_client = grpcclient.InferenceServerClient(
            url='localhost:8001')
    inputs = []
    outputs = []
    raw_input = np.fromfile('./data/dog.jpg', "uint8")
    image_data = np.expand_dims(raw_input, axis=0)
    inputs.append(grpcclient.InferInput('origin_img', image_data.shape, "UINT8"))
    outputs.append(grpcclient.InferRequestedOutput('yolo_infer_input'))
    inputs[0].set_data_from_numpy(image_data)
    print("Invoking inference...")
    model = 'yolo_preprocess'
    results = triton_client.infer(model_name=model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=10.0)   
    statistics = triton_client.get_inference_statistics(model_name=model)
    print(statistics)
    output = results.as_numpy('yolo_infer_input')
    with open('./data/preprocess_server_result.yaml', 'w') as f:
        yaml.dump(output.tolist(), f)

def test_preprocess_result_client():
    triton_client = grpcclient.InferenceServerClient(
            url='localhost:8001')
    inputs = []
    outputs = []
    raw_input = np.fromfile('./data/dog.jpg', "uint8")
    image_data = np.expand_dims(raw_input, axis=0)
    inputs.append(grpcclient.InferInput('origin_img', image_data.shape, "UINT8"))
    outputs.append(grpcclient.InferRequestedOutput('yolo_infer_input'))
    inputs[0].set_data_from_numpy(image_data)
    print("Invoking inference...")
    model = 'yolo_preprocess'
    results = triton_client.infer(model_name=model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=10.0)   
    statistics = triton_client.get_inference_statistics(model_name=model)
    print(statistics)

    print('===================================')
    print(results.as_numpy('yolo_infer_input').shape)
    print('===================================')
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('images', [1, 3, 640, 640], "FP32"))
    outputs.append(grpcclient.InferRequestedOutput('num_dets'))
    outputs.append(grpcclient.InferRequestedOutput('det_boxes'))
    outputs.append(grpcclient.InferRequestedOutput('det_scores'))
    outputs.append(grpcclient.InferRequestedOutput('det_classes'))
    inputs[0].set_data_from_numpy(results.as_numpy('yolo_infer_input'))
    print("Invoking inference...")
    model = 'yolo_trt'
    results = triton_client.infer(model_name=model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=10.0)   
    statistics = triton_client.get_inference_statistics(model_name=model)
    print(statistics)
    num_dets = results.as_numpy('num_dets')
    det_boxes = results.as_numpy('det_boxes')
    det_scores = results.as_numpy('det_scores')
    det_classes = results.as_numpy('det_classes')   
    input_image = cv2.imread('./data/dog.jpg') 
    detected_objects = postprocess(num_dets, det_boxes, det_scores, det_classes, input_image.shape[1], input_image.shape[0], [640, 640])
    #print(f"Detected objects: {len(detected_objects)}")

    for box in detected_objects:
        print(f"{COCOLabels(box.classID).name}: {box.confidence}")
        input_image = render_box(input_image, box.box(), color=tuple(RAND_COLORS[box.classID % 64].tolist()))
        size = get_text_size(input_image, f"{COCOLabels(box.classID).name}: {box.confidence:.2f}", normalised_scaling=0.6)
        input_image = render_filled_box(input_image, (box.x1 - 3, box.y1 - 3, box.x1 + size[0], box.y1 + size[1]), color=(220, 220, 220))
        input_image = render_text(input_image, f"{COCOLabels(box.classID).name}: {box.confidence:.2f}", (box.x1, box.y1), color=(30, 30, 30), normalised_scaling=0.5)

    cv2.imwrite('./data/test_preprocess_triton.png', input_image)
    print(f"Saved result to")



def test_postprocess_client():
    triton_client = grpcclient.InferenceServerClient(
            url='localhost:8001')
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('images', [1, 3, 640, 640], "FP32"))
    outputs.append(grpcclient.InferRequestedOutput('num_dets'))
    outputs.append(grpcclient.InferRequestedOutput('det_boxes'))
    outputs.append(grpcclient.InferRequestedOutput('det_scores'))
    outputs.append(grpcclient.InferRequestedOutput('det_classes'))

    input_image = cv2.imread('./data/dog.jpg') 
    input_image_buffer = preprocess(input_image, [640,640])
    input_image_buffer = np.expand_dims(input_image_buffer, axis=0)   
    inputs[0].set_data_from_numpy(input_image_buffer)
    print("Invoking inference...")
    model = 'yolo_trt'
    results = triton_client.infer(model_name=model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=10.0)   
    statistics = triton_client.get_inference_statistics(model_name=model)
    print(statistics)
    num_dets = results.as_numpy('num_dets')
    print('===============')
    print(num_dets)
    det_boxes = results.as_numpy('det_boxes')
    det_scores = results.as_numpy('det_scores')
    det_classes = results.as_numpy('det_classes')
    
    # num_dets = np.expand_dims(num_dets, axis=0)
    # det_boxes = np.expand_dims(det_boxes, axis=0)
    # det_scores = np.expand_dims(det_scores, axis=0)
    # det_classes = np.expand_dims(det_classes, axis=0)
    inputs_post = []
    outputs_post = []
    inputs_post.append(grpcclient.InferInput('num_dets', [1, 1], "INT32").set_data_from_numpy(num_dets))
    inputs_post.append(grpcclient.InferInput('det_boxes', [ 1, 100, 4], "FP32").set_data_from_numpy(det_boxes))
    inputs_post.append(grpcclient.InferInput('det_scores', [ 1, 100], "FP32").set_data_from_numpy(det_scores))
    inputs_post.append(grpcclient.InferInput('det_classes', [1, 100], "INT32").set_data_from_numpy(det_classes))

    
    outputs_post.append(grpcclient.InferRequestedOutput('final_boxes'))
    outputs_post.append(grpcclient.InferRequestedOutput('final_confidents'))
    outputs_post.append(grpcclient.InferRequestedOutput('class_label_id'))
    outputs_post.append(grpcclient.InferRequestedOutput('final_num_dets'))

    model = 'yolo_postprocess'
    results = triton_client.infer(model_name=model,
                                      inputs=inputs_post,
                                      outputs=outputs_post,
                                      client_timeout=10.0)   
    statistics = triton_client.get_inference_statistics(model_name=model)
    print(statistics)    

    final_boxes = results.as_numpy('final_boxes')
    final_confidents = results.as_numpy('final_confidents')
    class_label_id = results.as_numpy('class_label_id')
    final_num_dets = results.as_numpy('final_num_dets')
    

    # print(final_confidents)
    for i in range(final_num_dets[0][0]):

        print(f"{COCOLabels(class_label_id[0,i]).name}: {final_confidents[0,i]}")
        input_image = render_box(input_image, (final_boxes[0,i,0],final_boxes[0,i,1],final_boxes[0,i,2],final_boxes[0,i,3]),
        color=tuple(RAND_COLORS[class_label_id[0,i] % 64].tolist()))
        size = get_text_size(input_image, f"{COCOLabels(class_label_id[0,i]).name}: {final_confidents[0,i]:.2f}", normalised_scaling=0.6)
        input_image = render_filled_box(input_image, (final_boxes[0,i,0] - 3, final_boxes[0,i,1] - 3, final_boxes[0,i,0] + size[0], final_boxes[0,i,1] + size[1]), color=(220, 220, 220))
        input_image = render_text(input_image, f"{COCOLabels(class_label_id[0,i]).name}: {final_confidents[0,i]:.2f}",
        (final_boxes[0,i,0],final_boxes[0,i,1]), color=(30, 30, 30), normalised_scaling=0.5)

    cv2.imwrite('./data/test_postprocess_triton.png', input_image)
    print(f"Saved result to")


def lookup_service():
    channel = grpc.insecure_channel('localhost:8001')
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    # Metadata
    request = service_pb2.ServerMetadataRequest()
    response = grpc_stub.ServerMetadata(request)
    print("server metadata:\n{}".format(response))

    request = service_pb2.ModelMetadataRequest(name='yolo_postprocess', version='1')
    response = grpc_stub.ModelMetadata(request)
    print("model metadata:\n{}".format(response))

    # Configuration
    request = service_pb2.ModelConfigRequest(name='yolo_postprocess', version='1')
    response = grpc_stub.ModelConfig(request)
    print("model config:\n{}".format(response))

    
def test_yolo_end2end_client():
    triton_client = grpcclient.InferenceServerClient(
            url='localhost:8001')
    inputs = []
    outputs = []
    raw_input = cv2.imread('./data/dog.jpg')
    raw_input = raw_input.reshape(1, raw_input.shape[0], raw_input.shape[1], raw_input.shape[2])
    inputs.append(grpcclient.InferInput('yolo_input', raw_input.shape, "UINT8"))
    inputs[0].set_data_from_numpy(np.asarray(raw_input))
    outputs.append(grpcclient.InferRequestedOutput('yolo_boxes'))
    outputs.append(grpcclient.InferRequestedOutput('yolo_confidents'))
    outputs.append(grpcclient.InferRequestedOutput('yolo_class_label_id'))
    outputs.append(grpcclient.InferRequestedOutput('yolo_num_dets'))
    model = 'yolo_end2end'
    results = triton_client.infer(model_name=model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=10.0)   
    statistics = triton_client.get_inference_statistics(model_name=model)
    print(statistics)    
    final_boxes = results.as_numpy('yolo_boxes')
    final_confidents = results.as_numpy('yolo_confidents')
    class_label_id = results.as_numpy('yolo_class_label_id')
    final_num_dets = results.as_numpy('yolo_num_dets')
    
    print(final_boxes.shape)
    print(final_confidents.shape)
    print(class_label_id.shape)
    print(final_num_dets.shape)
    final_boxes = final_boxes[0]
    final_confidents = final_confidents[0]
    class_label_id = class_label_id[0]
    final_num_dets = final_num_dets[0]
    input_image = cv2.imread('./data/dog.jpg')
    # print(final_confidents)
    for i in range(final_num_dets[0][0]):

        print(f"{COCOLabels(class_label_id[0,i]).name}: {final_confidents[0,i]}")
        input_image = render_box(input_image, (final_boxes[0,i,0],final_boxes[0,i,1],final_boxes[0,i,2],final_boxes[0,i,3]),
        color=tuple(RAND_COLORS[class_label_id[0,i] % 64].tolist()))
        size = get_text_size(input_image, f"{COCOLabels(class_label_id[0,i]).name}: {final_confidents[0,i]:.2f}", normalised_scaling=0.6)
        input_image = render_filled_box(input_image, (final_boxes[0,i,0] - 3, final_boxes[0,i,1] - 3, final_boxes[0,i,0] + size[0], final_boxes[0,i,1] + size[1]), color=(220, 220, 220))
        input_image = render_text(input_image, f"{COCOLabels(class_label_id[0,i]).name}: {final_confidents[0,i]:.2f}",
        (final_boxes[0,i,0],final_boxes[0,i,1]), color=(30, 30, 30), normalised_scaling=0.5)

    cv2.imwrite('./data/test_end2end_triton.png', input_image)
    print(f"Saved result to")

def test_cv_serial():
    raw_input = cv2.imread('./data/dog.jpg')
    mat_data = raw_input.tobytes()
    image_data = np.frombuffer(mat_data, np.uint8)
    image_shape = (576,768,3)
    image_data = image_data.reshape(image_shape)
    opencv_mat = cv2.UMat(image_data)
    cv2.imwrite('./data/test_serial.jpg', opencv_mat)

    shape_info = np.array(raw_input.shape)
    print(shape_info.shape)





test_yolo_end2end_client()    


