import os
from argparse import ArgumentParser
from tqdm import tqdm
import cv2
import numpy as np
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection

def getfilename(path):
    file = os.path.split(path)[-1]
    name = os.path.splitext(file)[0]
    return name

class Post_back:
    def __init__(self, box, org_image) -> None:
        x1, y1, x2, y2 = box
        mask = np.zeros(org_image.shape, np.uint8)
        mask[y1:y2, x1:x2, :] = 255
        val = (x2 - x1) // 20
        erode_val = int(val * 1.5 / 2)
        erode_mask = cv2.erode(mask, np.ones((erode_val, erode_val), np.uint8), iterations=1)
        
        blur_mask = cv2.blur(erode_mask, (val, val))
        self.mask = cv2.bitwise_and(blur_mask, mask)
        # draw the mask
        # debugmask = cv2.polylines(self.mask, np.array([[(x1, y1), (x1, y2), (x2, y2), (x2, y1)]]), 1, [0, 0, 255])
        # cv2.imwrite('log/mask.jpg', debugmask)
        self.mask = self.mask / 255
        self.org_image = org_image

    def back_to_pic(self, croped_image):
        x1, y1, x2, y2 = box
        croped_image = cv2.resize(croped_image, (x2 - x1, y2 - y1))
        back_croped_image = np.zeros(self.org_image.shape, np.uint8)
        back_croped_image[y1:y2, x1:x2, :] = croped_image
        return (back_croped_image * self.mask + self.org_image * (1 - self.mask)).astype(np.uint8)

if not os.path.exists('log'):
    os.mkdir('log')

parser = ArgumentParser()
parser.add_argument("--source_image", default='data/boy.png', help="path to source image")
parser.add_argument("--driving_video", default='data/trump.mp4', help="path to driving video")
parser.add_argument("--result_video", default='0', help="path to output")
parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
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
        ymax = min(ymin + int(height * 1.2), h)
        ymin = max(ymin - int(height * 0.7), 0)
        height = ymax - ymin
        xmin = max(xmin - (height - width) // 2, 0)
        xmax = min(xmin + height, w)
        crop = image[ymin:ymax, xmin:xmax]
        cv2.imwrite(crop_path, crop)
        box = xmin, ymin, xmax, ymax
        break

driving_video = args.driving_video
source_image = crop_path
result = os.path.join('log', f"{getfilename(driving_video)}_{getfilename(args.source_image)}.mp4")
cmd = f"python3 demo.py  --config config/vox-256.yaml --driving_video {driving_video} " \
      f"--source_image {source_image} --checkpoint checkpoints/vox256.pth " \
      f"--result_video {result} --relative --adapt_scale --find_best_frame {'--cpu' if args.cpu else ' '}"
print(cmd)
os.system(cmd)


enhance_result = os.path.join('log', f"enhance_{getfilename(result)}.mp4")
cmd = f"cd Real-ESRGAN && python3 inference_video.py --model_path experiments/pretrained_models/RealESRGAN_x4plus.pth --input {os.path.abspath(result)} --output {os.path.abspath(enhance_result)}"
if args.face_enhance:
    cmd += " --face_enhance"
print(cmd)
os.system(cmd)

print("start post back")
video = cv2.VideoCapture(enhance_result)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = video.get(5)
frame_cnt = int(video.get(7))
height, width, _ = image.shape
final_result = os.path.join('log', f"final_{getfilename(result)}.mp4") if args.result_video == '0' else args.result_video
writer = cv2.VideoWriter(final_result, fourcc, fps, (width, height))
in_while = False
post_back = Post_back(box, image)
for i in tqdm(range(frame_cnt)):
    ret, img = video.read()
    if not ret:
        break
    writer.write(post_back.back_to_pic(img))
    in_while = True
writer.release()
print('\n\nsuccess' if in_while else "\n\nfail")