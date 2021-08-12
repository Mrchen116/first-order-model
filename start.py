import os
from argparse import ArgumentParser
import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection

def getfilename(path):
    file = os.path.split(path)
    name = os.path.splitext(file)
    return name

parser = ArgumentParser()
parser.add_argument("--source_image", default='data/face_boy.png', help="path to source image")
parser.add_argument("--driving_video", default='data/trump.mp4', help="path to driving video")
parser.add_argument("--result_video", default='0', help="path to output")
parser.add_argument("--cutted", action="store_true", help="if the iamge has been cutted")

args = parser.parse_args()

if not args.cutted:
    with mp_face_detection.FaceDetection(
        model_selection=1) as face_detection:
        image = cv2.imread(args.source_image)
        result = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        assert result.detections
        for detection in result.detections:
            print(detection.xmin)

exit(0)
driving_video = args.driving_video
source_image = args.source_image
result = os.path.join('log', f"{getfilename(driving_video)}_{getfilename(source_image)}.mp4") if args.result_video == '0' else args.result_video
cmd = f"python demo.py  --config config/vox-256.yaml --driving_video {driving_video} " \
      f"--source_image {source_image} --checkpoint checkpoints/vox256.pth --relative --adapt_scale"
print(cmd)
os.system(cmd)