import io
import json

import numpy as np
import cv2
from PIL import Image
import triton_python_backend_utils as pb_utils
class BoundingBox:
    def __init__(self, classID, confidence, x1, x2, y1, y2, image_width, image_height):
        self.classID = classID
        self.confidence = confidence
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.u1 = x1 / image_width
        self.u2 = x2 / image_width
        self.v1 = y1 / image_height
        self.v2 = y2 / image_height

    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def center_absolute(self):
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    def center_normalized(self):
        return (0.5 * (self.u1 + self.u2), 0.5 * (self.v1 + self.v2))

    def size_absolute(self):
        return (self.x2 - self.x1, self.y2 - self.y1)

    def size_normalized(self):
        return (self.u2 - self.u1, self.v2 - self.v1)

def postprocess(num_dets, det_boxes, det_scores, det_classes, img_w, img_h, input_shape, letter_box=True):
    boxes = det_boxes[0, :num_dets[0][0]] / np.array([input_shape[0], input_shape[1], input_shape[0], input_shape[1]], dtype=np.float32)
    scores = det_scores[0, :num_dets[0][0]]
    classes = det_classes[0, :num_dets[0][0]].astype(int)

    old_h, old_w = img_h, img_w
    offset_h, offset_w = 0, 0
    if letter_box:
        if (img_w / input_shape[1]) >= (img_h / input_shape[0]):
            old_h = int(input_shape[0] * img_w / input_shape[1])
            offset_h = (old_h - img_h) // 2
        else:
            old_w = int(input_shape[1] * img_h / input_shape[0])
            offset_w = (old_w - img_w) // 2

    boxes = boxes * np.array([old_w, old_h, old_w, old_h], dtype=np.float32)
    if letter_box:
        boxes -= np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
    boxes = boxes.astype(int)

    detected_objects = []
    for box, score, label in zip(boxes, scores, classes):
        detected_objects.append(BoundingBox(label, score, box[0], box[2], box[1], box[3], img_w, img_h))
    return detected_objects

class TritonPythonModel:
    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "final_boxes")
        output1_config = pb_utils.get_output_config_by_name(model_config, "final_confidents")
        output2_config = pb_utils.get_output_config_by_name(model_config, "class_label_id")
        output3_config = pb_utils.get_output_config_by_name(model_config, "final_num_dets")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config["data_type"]
        )  
        self.output2_dtype = pb_utils.triton_string_to_numpy(
            output2_config["data_type"]
        )    
        self.output3_dtype = pb_utils.triton_string_to_numpy(
            output3_config["data_type"]
        )   

    def execute(self, requests):
        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype
        output2_dtype = self.output2_dtype
        output3_dtype = self.output3_dtype

        responses = []

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "num_dets_in")
            in_1 = pb_utils.get_input_tensor_by_name(request, "det_boxes_in")
            in_2 = pb_utils.get_input_tensor_by_name(request, "det_scores_in")
            in_3 = pb_utils.get_input_tensor_by_name(request, "det_classes_in")
            in_4 = pb_utils.get_input_tensor_by_name(request, "origin_img_shape")
            
            
            num_dets = in_0.as_numpy()
            det_boxes = in_1.as_numpy()
            det_scores = in_2.as_numpy()
            det_classes = in_3.as_numpy()
            img_shape = in_4.as_numpy()
            # print(img_shape)
            detected_objects = postprocess(num_dets, det_boxes, det_scores, det_classes,img_shape[0][2],img_shape[0][1] , [640, 640])
            
            final_boxes = np.zeros((1,100,4), dtype = output0_dtype)
            final_confs = np.zeros((1,100), dtype = output1_dtype)
            class_label = np.zeros((1,100), dtype = output2_dtype)
            final_num_dets = np.zeros((1,1), dtype = output3_dtype)
            final_num_dets[0][0] = num_dets[0][0]
            for i in range(len(detected_objects)):
                final_boxes[0, i, :] = [detected_objects[i].x1, detected_objects[i].y1, detected_objects[i].x2, detected_objects[i].y2]
                final_confs[0,i] = detected_objects[i].confidence
                class_label[0,i] = detected_objects[i].classID
            out_tensor_0 = pb_utils.Tensor("final_boxes", final_boxes)
            out_tensor_1 = pb_utils.Tensor("final_confidents", final_confs)
            out_tensor_2 = pb_utils.Tensor("class_label_id", class_label)
            out_tensor_3 = pb_utils.Tensor("final_num_dets", final_num_dets)


            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1, out_tensor_2, out_tensor_3]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")