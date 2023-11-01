import io
import json

import numpy as np
import cv2
from PIL import Image
import triton_python_backend_utils as pb_utils
from typing import List, Tuple

def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    # get maximum value locations
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    # get maximum value across x and y axis
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    # reshape
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals


def decode(simcc_x: np.ndarray, simcc_y: np.ndarray,
           simcc_split_ratio) -> Tuple[np.ndarray, np.ndarray]:
    """Modulate simcc distribution with Gaussian.

    Args:
        simcc_x (np.ndarray[K, Wx]): model predicted simcc in x.
        simcc_y (np.ndarray[K, Wy]): model predicted simcc in y.
        simcc_split_ratio (int): The split ratio of simcc.

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: keypoints in shape (K, 2) or (n, K, 2)
        - np.ndarray[float32]: scores in shape (K,) or (n, K)
    """
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio

    return keypoints, scores

def postprocess(outputs: List[np.ndarray],
                model_input_size: Tuple[int, int],
                center: Tuple[int, int],
                scale: Tuple[int, int],
                simcc_split_ratio: float = 2.0
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Postprocess for RTMPose model output.

    Args:
        outputs (np.ndarray): Output of RTMPose model.
        model_input_size (tuple): RTMPose model Input image size.
        center (tuple): Center of bbox in shape (x, y).
        scale (tuple): Scale of bbox in shape (w, h).
        simcc_split_ratio (float): Split ratio of simcc.

    Returns:
        tuple:
        - keypoints (np.ndarray): Rescaled keypoints.
        - scores (np.ndarray): Model predict scores.
    """
    # use simcc to decode
    simcc_x, simcc_y = outputs
    keypoints, scores = decode(simcc_x, simcc_y, simcc_split_ratio)

    # rescale keypoints
    keypoints = keypoints / model_input_size * scale + center - scale / 2

    return keypoints, scores

class TritonPythonModel:
    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "det_keypoints")
        output1_config = pb_utils.get_output_config_by_name(model_config, "det_scores")

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
            in_0 = pb_utils.get_input_tensor_by_name(request, "model_simcc_x")
            in_1 = pb_utils.get_input_tensor_by_name(request, "model_simcc_y")
            in_2 = pb_utils.get_input_tensor_by_name(request, "image_center")
            in_3 = pb_utils.get_input_tensor_by_name(request, "image_scale")
            
            simcc_x = in_0.as_numpy()
            simcc_y = in_1.as_numpy()
            center = in_2.as_numpy()
            scale = in_3.as_numpy()
            
            model_input_size = (288, 384)
  
            keypoints, scores = postprocess([simcc_x, simcc_y], model_input_size, center[0], scale[0])


            out_tensor_0 = pb_utils.Tensor("det_keypoints", keypoints.astype(output0_dtype))
            out_tensor_1 = pb_utils.Tensor("det_scores", scores.astype(output1_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0,out_tensor_1]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")