# import the necessary packages
import argparse
from face_detector import FaceDetector
from webcam_video_feed import WebcamVideoFeed

# construct the argument parse and parse the arguments
def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", 
        default="deploy.prototxt.txt",
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", 
        default="res10_300x300_ssd_iter_140000.caffemodel",
        help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    return args["prototxt"], args["model"], args["confidence"]

prototxt, model, min_confidence = parse_arguments()

fd = FaceDetector(prototxt, model, min_confidence)
vf = WebcamVideoFeed(20, frame_worker=fd.detect)
vf.start()