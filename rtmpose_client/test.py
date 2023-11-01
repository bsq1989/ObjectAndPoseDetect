# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
from typing import List, Tuple

import cv2
import numpy as np

import grpc
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from tritonclient.grpc import service_pb2, service_pb2_grpc

def show_kpt(keypoints,img):#rgb
    rules = [
        {'srt_kpt_id': 19, 'dst_kpt_id': 7, 'color': [255,0,255], 'thickness': 2},
        {'srt_kpt_id': 7, 'dst_kpt_id': 9, 'color': [255,0,255], 'thickness': 2},
        {'srt_kpt_id': 9, 'dst_kpt_id': 11, 'color': [255,0,255], 'thickness': 2},
        {'srt_kpt_id': 6, 'dst_kpt_id': 7, 'color': [0,255,0],'thickness': 2},
        {'srt_kpt_id': 19, 'dst_kpt_id': 6, 'color': [255,165,0], 'thickness': 2},
        {'srt_kpt_id': 6, 'dst_kpt_id': 8, 'color': [255,165,0], 'thickness': 2},
        {'srt_kpt_id': 8, 'dst_kpt_id': 10, 'color': [255,165,0], 'thickness': 2},
        {'srt_kpt_id': 7, 'dst_kpt_id': 13, 'color': [0,255,0], 'thickness': 2},
        {'srt_kpt_id': 6, 'dst_kpt_id': 12, 'color': [0,255,0], 'thickness': 2},
        {'srt_kpt_id': 12, 'dst_kpt_id': 13, 'color': [0,255,0], 'thickness': 2},
        {'srt_kpt_id': 13, 'dst_kpt_id': 15, 'color': [255,255,0], 'thickness': 2},
        {'srt_kpt_id': 12, 'dst_kpt_id': 14, 'color': [0,255,255], 'thickness': 2},
        {'srt_kpt_id': 15, 'dst_kpt_id': 17, 'color': [255,255,0], 'thickness': 2},
        {'srt_kpt_id': 14, 'dst_kpt_id': 16, 'color': [0,255,255], 'thickness': 2},
        {'srt_kpt_id': 17, 'dst_kpt_id': 22, 'color': [255,255,0], 'thickness': 2},
        {'srt_kpt_id': 17, 'dst_kpt_id': 26, 'color': [255,255,0], 'thickness': 2},
        {'srt_kpt_id': 22, 'dst_kpt_id': 26, 'color': [255,255,0], 'thickness': 2},
        {'srt_kpt_id': 25, 'dst_kpt_id': 21, 'color': [0,255,255],'thickness': 2},
        {'srt_kpt_id': 16, 'dst_kpt_id': 25, 'color': [0,255,255], 'thickness': 2},
        {'srt_kpt_id': 16, 'dst_kpt_id': 21, 'color': [0,255,255], 'thickness': 2}
    ]
    for rule in rules:
        srt_kpt_id=rule["srt_kpt_id"]
        dst_kpt_id = rule["dst_kpt_id"]
        color= rule["color"]
        thickness=rule["thickness"]
        x1 = int(keypoints[0][srt_kpt_id - 1][0])
        y1 = int(keypoints[0][srt_kpt_id - 1][1])
        x2 = int(keypoints[0][dst_kpt_id - 1][0])
        y2 = int(keypoints[0][dst_kpt_id - 1][1])
        cv2.line(img, (x1,y1), (x2,y2), color, thickness)

    # cv2.imshow("1",img)
    # cv2.waitKey(1000)
    # print()

def show_points(keypoints, img):
    custom_options = {
        'point_indices': [5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,24,25],
        'point_color': (0, 0, 255),
        'point_radius': 3
    }

    point_indices = custom_options.get('point_indices', None)
    point_color = custom_options.get('point_color', (0, 0, 255))
    point_radius = custom_options.get('point_radius', 3)

    if point_indices is None:
        point_indices = range(len(keypoints))

    for idx in point_indices:
        if 0 <= idx < keypoints.shape[1]:
            x, y = keypoints[0][idx]
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), point_radius, point_color, -1)  # -1 fills the circle



def preprocess(
    img: np.ndarray, input_size: Tuple[int, int] = (192, 256)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Do preprocessing for RTMPose model inference.

    Args:
        img (np.ndarray): Input image in shape.
        input_size (tuple): Input image size in shape (w, h).

    Returns:
        tuple:
        - resized_img (np.ndarray): Preprocessed image.
        - center (np.ndarray): Center of image.
        - scale (np.ndarray): Scale of image.
    """
    # get shape of image
    img_shape = img.shape[:2]
    bbox = np.array([0, 0, img_shape[1], img_shape[0]])

    # get center and scale
    center, scale = bbox_xyxy2cs(bbox, padding=1.25)

    # do affine transformation
    resized_img, scale = top_down_affine(input_size, scale, center, img)

    # normalize image
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    resized_img = (resized_img - mean) / std

    resized_img = resized_img.transpose((2, 0, 1)).astype(np.float32)

    return resized_img, center, scale





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


def visualize(img: np.ndarray,
              keypoints: np.ndarray,
              scores: np.ndarray,
              filename: str = 'output.jpg',
              thr=0.3) -> np.ndarray:
    """Visualize the keypoints and skeleton on image.

    Args:
        img (np.ndarray): Input image in shape.
        keypoints (np.ndarray): Keypoints in image.
        scores (np.ndarray): Model predict scores.
        thr (float): Threshold for visualize.

    Returns:
        img (np.ndarray): Visualized image.
    """
    # default color
    skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17),
                (15, 18), (15, 19), (16, 20), (16, 21), (16, 22), (91, 92),
                (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98),
                (98, 99), (91, 100), (100, 101), (101, 102), (102, 103),
                (91, 104), (104, 105), (105, 106), (106, 107), (91, 108),
                (108, 109), (109, 110), (110, 111), (112, 113), (113, 114),
                (114, 115), (115, 116), (112, 117), (117, 118), (118, 119),
                (119, 120), (112, 121), (121, 122), (122, 123), (123, 124),
                (112, 125), (125, 126), (126, 127), (127, 128), (112, 129),
                (129, 130), (130, 131), (131, 132)]
    palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
               [255, 153, 255], [102, 178, 255], [255, 51, 51]]
    link_color = [
        1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2,
        2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 2, 2, 2,
        2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
    ]
    point_color = [
        0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
        4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 3, 2, 2, 2, 2, 4, 4, 4,
        4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
    ]

    # draw keypoints and skeleton
    for kpts, score in zip(keypoints, scores):
        for kpt, color in zip(kpts, point_color):
            cv2.circle(img, tuple(kpt.astype(np.int32)), 1, palette[color], 1,
                       cv2.LINE_AA)
        # for (u, v), color in zip(skeleton, link_color):
        #     if score[u] > thr and score[v] > thr:
        #         cv2.line(img, tuple(kpts[u].astype(np.int32)),
        #                  tuple(kpts[v].astype(np.int32)), palette[color], 2,
        #                  cv2.LINE_AA)

    # save to local
    cv2.imwrite(filename, img)

    return img


def bbox_xyxy2cs(bbox: np.ndarray,
                 padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (left, top, right, bottom)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """
    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    # get bbox center and scale
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def _fix_aspect_ratio(bbox_scale: np.ndarray,
                      aspect_ratio: float) -> np.ndarray:
    """Extend the scale to match the given aspect ratio.

    Args:
        scale (np.ndarray): The image scale (w, h) in shape (2, )
        aspect_ratio (float): The ratio of ``w/h``

    Returns:
        np.ndarray: The reshaped image scale in (2, )
    """
    w, h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(w > h * aspect_ratio,
                          np.hstack([w, w / aspect_ratio]),
                          np.hstack([h * aspect_ratio, h]))
    return bbox_scale


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c


def get_warp_matrix(center: np.ndarray,
                    scale: np.ndarray,
                    rot: float,
                    output_size: Tuple[int, int],
                    shift: Tuple[float, float] = (0., 0.),
                    inv: bool = False) -> np.ndarray:
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # compute transformation matrix
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    # get four corners of the src rectangle in the original image
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    # get four corners of the dst rectangle in the input image
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return warp_mat


def top_down_affine(input_size: dict, bbox_scale: dict, bbox_center: dict,
                    img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the bbox image as the model input by affine transform.

    Args:
        input_size (dict): The input size of the model.
        bbox_scale (dict): The bbox scale of the img.
        bbox_center (dict): The bbox center of the img.
        img (np.ndarray): The original image.

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: img after affine transform.
        - np.ndarray[float32]: bbox scale after affine transform.
    """
    w, h = input_size
    warp_size = (int(w), int(h))

    # reshape bbox to fixed aspect ratio
    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)

    # get the affine matrix
    center = bbox_center
    scale = bbox_scale
    rot = 0
    warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

    # do affine transform
    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return img, bbox_scale


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


def test_preprocess_client():
    triton_client = grpcclient.InferenceServerClient(
            url='localhost:8001')
    inputs = []
    outputs = []
    raw_input = np.fromfile('./data/human-pose.jpeg', "uint8")
    image_data = np.expand_dims(raw_input, axis=0)
    inputs.append(grpcclient.InferInput('origin_img', image_data.shape, "UINT8"))
    outputs.append(grpcclient.InferRequestedOutput('rtmpose_infer_input'))
    outputs.append(grpcclient.InferRequestedOutput('input_center'))
    outputs.append(grpcclient.InferRequestedOutput('input_scale'))
    inputs[0].set_data_from_numpy(image_data)
    print("Invoking inference...")
    model = 'rtmpose_preprocess'
    results = triton_client.infer(model_name=model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=10.0)   
    statistics = triton_client.get_inference_statistics(model_name=model)
    print(statistics)

    
    center = results.as_numpy('input_center')
    scale = results.as_numpy('input_scale')


    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('input', [1, 3, 384, 288], "FP32"))
    outputs.append(grpcclient.InferRequestedOutput('simcc_x'))
    outputs.append(grpcclient.InferRequestedOutput('simcc_y'))

    inputs[0].set_data_from_numpy(results.as_numpy('rtmpose_infer_input'))

    model = 'rtmpose_trt'
    results = triton_client.infer(model_name=model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=10.0)   
    statistics = triton_client.get_inference_statistics(model_name=model)
    print(statistics)    

    inputs = []
    outputs = []

    inputs.append(grpcclient.InferInput('model_simcc_x', [1, 26, 576], "FP32"))
    inputs.append(grpcclient.InferInput('model_simcc_y', [1, 26, 768], "FP32"))
    inputs.append(grpcclient.InferInput('image_center', [1, 2], "FP32"))
    inputs.append(grpcclient.InferInput('image_scale', [1, 2], "FP32"))

    outputs.append(grpcclient.InferRequestedOutput('det_keypoints'))
    outputs.append(grpcclient.InferRequestedOutput('det_scores'))
    print(results.as_numpy('simcc_x').shape)
    inputs[0].set_data_from_numpy(results.as_numpy('simcc_x'))
    inputs[1].set_data_from_numpy(results.as_numpy('simcc_y'))
    inputs[2].set_data_from_numpy(center)
    inputs[3].set_data_from_numpy(scale)    


    model = 'rtmpose_postprocess'
    results = triton_client.infer(model_name=model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=10.0)   
    statistics = triton_client.get_inference_statistics(model_name=model)
    print(statistics)    

    # keypoints, scores = postprocess([simcc_x, simcc_y], model_input_size, center[0], scale[0])
    # print('===========================')
    # print(keypoints.shape)
    # print(scores.shape)

    keypoints = results.as_numpy('det_keypoints')
    scores = results.as_numpy('det_scores')

    print(keypoints.shape)
    print(keypoints.dtype)

    print(scores.shape)
    print(scores.dtype)
    
    cv_input = cv2.imread('./data/human-pose.jpeg')
    show_kpt(keypoints, cv_input)

    show_points(keypoints, cv_input)
    cv2.imwrite('./data/hp_postprocess_triton_infer.jpg', cv_input)



def test_client():
    triton_client = grpcclient.InferenceServerClient(
            url='localhost:8001')
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('input', [1, 3, 384, 288], "FP32"))
    outputs.append(grpcclient.InferRequestedOutput('simcc_x'))
    outputs.append(grpcclient.InferRequestedOutput('simcc_y'))
    input_image = cv2.imread('./data/human-pose.jpeg')

    model_input_size = (288, 384)
    resized_img, center, scale = preprocess(input_image, model_input_size)

    input_image_buffer = np.expand_dims(resized_img, axis=0)   
    inputs[0].set_data_from_numpy(input_image_buffer)

    model = 'rtmpose_trt'
    results = triton_client.infer(model_name=model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=10.0)   
    statistics = triton_client.get_inference_statistics(model_name=model)
    print(statistics)    

    simcc_x = results.as_numpy('simcc_x')
    simcc_y = results.as_numpy('simcc_y')
    # print('===========================')
    # print(simcc_x.shape)
    # print(simcc_y.shape)

    keypoints, scores = postprocess([simcc_x, simcc_y], model_input_size, center, scale)
    # print('===========================')
    # print(keypoints.shape)
    # print(scores.shape)

    show_kpt(keypoints, input_image)

    show_points(keypoints, input_image)
    cv2.imwrite('./data/hp_triton_infer.jpg', input_image)

def lookup_service():
    channel = grpc.insecure_channel('localhost:8001')
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    # Metadata
    request = service_pb2.ServerMetadataRequest()
    response = grpc_stub.ServerMetadata(request)
    print("server metadata:\n{}".format(response))

    request = service_pb2.ModelMetadataRequest(name='rtmpose_preprocess', version='1')
    response = grpc_stub.ModelMetadata(request)
    print("model metadata:\n{}".format(response))

    # Configuration
    request = service_pb2.ModelConfigRequest(name='rtmpose_preprocess', version='1')
    response = grpc_stub.ModelConfig(request)
    print("model config:\n{}".format(response))

def main():
    args = parse_args()
    logger.info('Start running model on RTMPose...')

    # read image from file
    logger.info('1. Read image from {}...'.format(args.image_file))
    img = cv2.imread(args.image_file)

    # build onnx model
    logger.info('2. Build onnx model from {}...'.format(args.onnx_file))
    sess = build_session(args.onnx_file, args.device)
    h, w = sess.get_inputs()[0].shape[2:]
    model_input_size = (w, h)

    # preprocessing
    logger.info('3. Preprocess image...')
    resized_img, center, scale = preprocess(img, model_input_size)

    # inference
    logger.info('4. Inference...')
    start_time = time.time()
    outputs = inference(sess, resized_img)
    end_time = time.time()
    logger.info('4. Inference done, time cost: {:.4f}s'.format(end_time -
                                                               start_time))

    # postprocessing
    logger.info('5. Postprocess...')
    keypoints, scores = postprocess(outputs, model_input_size, center, scale)

    # visualize inference result
    logger.info('6. Visualize inference result...')
    visualize(img, keypoints, scores, args.save_path)

    logger.info('Done...')

def test_rtmpose_end2end():
    triton_client = grpcclient.InferenceServerClient(
            url='localhost:8001')
    inputs = []
    outputs = []
    # raw_input = np.fromfile('./data/human-pose.jpeg', "uint8")
    # image_data = np.expand_dims(raw_input, axis=0)
    image_data = cv2.imread('./data/human-pose.jpeg')
    image_data = image_data.reshape(1, image_data.shape[0], image_data.shape[1], image_data.shape[2])
    inputs.append(grpcclient.InferInput('rtmpose_input', image_data.shape, "UINT8"))
    outputs.append(grpcclient.InferRequestedOutput('rtmpose_keypoints'))
    outputs.append(grpcclient.InferRequestedOutput('rtmpose_scores'))
    inputs[0].set_data_from_numpy(image_data)
    print("Invoking inference...")
    model = 'rtmpose_end2end'
    results = triton_client.infer(model_name=model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=10.0)   
    statistics = triton_client.get_inference_statistics(model_name=model)
    print(statistics)

    keypoints = results.as_numpy('rtmpose_keypoints')
    scores = results.as_numpy('rtmpose_scores')
    print(keypoints.shape)


    input_image = cv2.imread('./data/human-pose.jpeg')

    show_kpt(keypoints[0], input_image)

    show_points(keypoints[0], input_image)
    cv2.imwrite('./data/hp_triton_end2end.jpg', input_image)



if __name__ == '__main__':
    test_rtmpose_end2end()
