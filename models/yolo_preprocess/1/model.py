import io
import json

import numpy as np
import cv2
from PIL import Image
import triton_python_backend_utils as pb_utils
def preprocess(img, input_shape, letter_box=True):
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img

class TritonPythonModel:
    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "yolo_infer_input")
        output1_config = pb_utils.get_output_config_by_name(model_config, "origin_img_shape")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )  
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config["data_type"]
        )        

    def execute(self, requests):
        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype

        responses = []

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "origin_img")
            in_1 = pb_utils.get_input_tensor_by_name(request, "origin_shape")
            img_buffer = in_0.as_numpy()
            input_dim = np.asarray(img_buffer.shape).reshape(1,4)
            # print(img_buffer.shape)
            # img_shape = in_1.as_numpy()
            # img_buffer = img_buffer[0].reshape(img_shape[0])
            # image = Image.open(io.BytesIO(img.tobytes()))
            # cv2_img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
            cv2_img = img_buffer[0]
            pre_out = preprocess(cv2_img, [640,640])
            pre_out = np.expand_dims(pre_out, axis=0) 
            out_tensor_0 = pb_utils.Tensor("yolo_infer_input", pre_out.astype(output0_dtype))
            out_tensor_1 = pb_utils.Tensor("origin_img_shape", input_dim.astype(output1_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")