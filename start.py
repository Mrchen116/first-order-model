import os
from argparse import ArgumentParser
import cv2
import numpy as np
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection

def getfilename(path):
    file = os.path.split(path)[-1]
    name = os.path.splitext(file)[0]
    return name

def back_to_pic(croped_image, box, org_image):
    x1, y1, x2, y2 = box
    croped_image = cv2.resize(croped_image, (x2 - x1, y2 - y1))
    mask = np.zeros(org_image.shape, np.uint8)
    back_croped_image = mask.copy()
    back_croped_image[y1:y2, x1:x2, :] = croped_image
    mask[y1:y2, x1:x2, :] = 255
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=5)
    mask = cv2.blur(mask, (25, 25)) / 255
    # mask //= 255
    return (back_croped_image * mask + org_image * (1 - mask)).astype(np.uint8)

if not os.path.exists('log'):
    os.mkdir('log')

parser = ArgumentParser()
parser.add_argument("--source_image", default='data/boy.png', help="path to source image")
parser.add_argument("--driving_video", default='data/trump.mp4', help="path to driving video")
parser.add_argument("--result_video", default='0', help="path to output")
parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
# parser.add_argument("--cutted", action="store_true", help="if the iamge has been cutted")

args = parser.parse_args()
crop_path = os.path.join('log', 'cut_' + getfilename(args.source_image) + '.png')
image = cv2.imread(args.source_image)
h, w, _ = image.shape
with mp_face_detection.FaceDetection() as face_detection:
    result = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    assert result.detections
    for detection in result.detections:
        box = detection.location_data.relative_bounding_box
        xmin, ymin, width, height = int(box.xmin * w), int(box.ymin * h), int(box.width * w), int(box.height * h)
        ymax = min(ymin + int(height * 1.5), h)
        ymin = max(ymin - int(height * 0.8), 0)
        height = ymax - ymin
        xmin = max(xmin - (height - width) // 2, 0)
        xmax = min(xmin + height, w)
        crop = image[ymin:ymax, xmin:xmax]
        cv2.imwrite(crop_path, crop)
        box = xmin, ymin, xmax, ymax
        break

driving_video = args.driving_video
source_image = crop_path
result = os.path.join('log', f"{getfilename(driving_video)}_{getfilename(source_image)}.mp4") if args.result_video == '0' else args.result_video
cmd = f"python3 demo.py  --config config/vox-256.yaml --driving_video {driving_video} " \
      f"--source_image {source_image} --checkpoint checkpoints/vox256.pth " \
      f"--result_video {result} --relative --adapt_scale --find_best_frame {'--cpu' if args.cpu else ' '}"
print(cmd)
os.system(cmd)

video = cv2.VideoCapture(result)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 25
height, width, _ = image.shape
writer = cv2.VideoWriter(os.path.join('log', f"deal_{getfilename(result)}.mp4"), fourcc, fps, (width, height))
while True:
    ret, img = video.read()
    if not ret:
        break
    writer.write(back_to_pic(img, box, image))
writer.release()
print('\n\nsuccess')