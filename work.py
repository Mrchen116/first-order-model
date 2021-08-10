import os
import numpy as np
import cv2
import imutils
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths

def separate(img, mask):
    mask = np.repeat(mask, 3, axis=2)
    fg = img * mask
    bg = img * (1 - mask)
    return fg.astype(np.uint8), bg.astype(np.uint8)

IMG_PATH = 'data/boy.png'
IMG_RESIZE_PATH = 'log/resize.png'
VIDEO_PATH = 'data/trump.mp4'
FG_PATH = 'log/fg.png'
BG_PATH = 'log/bg.png'
CROP_RES_PATH = 'log/crop_result.mp4'

origin = cv2.imread(IMG_PATH)
resized = imutils.resize(origin, width=256)
cv2.imwrite(IMG_RESIZE_PATH, resized)

bodypix_model = load_model(download_model(
    BodyPixModelPaths.MOBILENET_FLOAT_100_STRIDE_8
))

image = tf.keras.preprocessing.image.load_img(IMG_RESIZE_PATH)
image_array = tf.keras.preprocessing.image.img_to_array(image)
result = bodypix_model.predict_single(image_array)
mask = result.get_mask(threshold=0.75)

face_mask = result.get_part_mask(mask, ['left_face', 'right_face'])
tf.keras.preprocessing.image.save_img(
    'log/face-mask.jpg',
    face_mask
)

crop = resized[:256, :256]
crop_mask = face_mask[:256, :256]

fg, bg = separate(crop, crop_mask)
cv2.imwrite(FG_PATH, fg)
cv2.imwrite(BG_PATH, bg)

cmd = "python demo.py  --config config/vox-256.yaml" \
      f" --driving_video {VIDEO_PATH}" \
      f" --source_image {FG_PATH}" \
      " --checkpoint checkpoints/vox-256.pth" \
      f" --result_video {CROP_RES_PATH}" \
      " --relative --adapt_scale"
print(cmd)
os.system(cmd)

