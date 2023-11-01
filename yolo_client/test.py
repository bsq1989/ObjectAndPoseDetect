from processing import preprocess
import cv2
import numpy as np
from PIL import Image
import io
import yaml
# image preprocess
input_image = cv2.imread('./data/dog.jpg')
resized_img = preprocess(input_image,[640,640])
with open('./data/preprocess_local_result.yaml', 'w') as f:
    yaml.dump(resized_img.tolist(), f)
# cv2.imwrite('./data/resize.jpg',resized_img)


# def test_input_serial():
# test_img = cv2.imread('./data/dog.jpg')
# buffer = np.fromfile('./data/dog.jpg', dtype="uint8")
# image = Image.open(io.BytesIO(buffer.tobytes()))
# cv2_img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

# cv2.imwrite('./data/test_serial.jpg', cv2_img)