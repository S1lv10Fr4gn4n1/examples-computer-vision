# https://www.pyimagesearch.com/2017/08/21/deep-learning-with-opencv/
import numpy as np
import argparse
import time
import cv2
import matplotlib.pyplot as plt

# Download bvlc_googlenet.caffemodel from https://github.com/BVLC/caffe/wiki/Model-Zoo

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="images/eagle.png", help="path to input image")
ap.add_argument("-p", "--prototxt", default="bvlc_googlenet.prototxt", help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="bvlc_googlenet.caffemodel", help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", default="synset_words.txt", help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())

image_path = args["image"]
prototxt_path = args["prototxt"]
model_path = args["model"]
labels_path = args["labels"]

# load the input image from disk
image = cv2.imread(image_path)

# load the class labels from disk
rows = open(labels_path).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# our CNN requires fixed spatial dimensions for our input image(s)
# so we need to ensure it is resized to 224x224 pixels while
# performing mean subtraction (104, 117, 123) to normalize the input;
# after executing this command our "blob" now has the shape:
# (1, 3, 224, 224)
blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

# load our serialied model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# set the blob as input to the network and perform a forward-pass to
# obtain our output classification
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] classification took {:.5} seconds".format(end - start))

# sort the indexes of the probabilities in descending order (higher
# probabilitiy first) and grab the top-5 predictions
idxs = np.argsort(preds[0])[::-1][:5]

# loop over the top-5 predictions and display them
for (i, idx) in enumerate(idxs):
    # draw the top prediction on the input image
    if i == 0:
        text = "Label: {}, {:.2f}%".format(classes[idx], preds[0][idx] * 100)
        cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # display the predicted label + associated probability to the console
    print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1, classes[idx], preds[0][idx]))

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()