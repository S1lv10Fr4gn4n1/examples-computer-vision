# https://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras/

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os
from datetime import datetime
from multiprocessing import Pool

t_start = datetime.now()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True,	help="path to output model file")
args = vars(ap.parse_args())

# dataset can be downloaded from https://www.kaggle.com/c/dogs-vs-cats/data
dataset_path = args["dataset"]
model_path = args["model"]

# grab the list of images that we'll be describing
print("[INFO] describing images...")
image_paths = list(paths.list_images(dataset_path))
 
# initialize the data matrix and labels list
def extra_feature_and_label(image_path):
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(image_path)
    label = image_path.split(os.path.sep)[-1].split(".")[0]
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    features = cv2.resize(image, (32,32)).flatten()
    return (features, label)

number_of_threads = os.cpu_count()
print("[INFO] processing images using {} threads".format(number_of_threads))
pool = Pool(number_of_threads)
results = pool.map(extra_feature_and_label, image_paths)
pool.close()
pool.join()

# construct a feature vector raw pixel intensities, then update
# the data matrix and labels list
labels = [l for (f, l) in results]
data = [f for (f, l) in results]

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# scale the input image pixels to the range [0, 1], then transform
# the labels into vectors in the range [0, num_classes] -- this
# generates a vector for each label where the index of the label
# is set to `1` and all other entries to `0`
data = np.array(data) / 255.0
labels = np_utils.to_categorical(labels, 2)

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(train_data, test_data, train_labels, test_labels) = train_test_split(data, labels, test_size=0.25, random_state=42)

# define the architecture of the network
model = Sequential()
model.add(Dense(768, input_dim=3072, init="uniform", activation="relu"))
model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
model.add(Dense(2))
model.add(Activation("softmax"))

# train the model using SGD
print("[INFO] compiling model...")
sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(train_data, train_labels, epochs=50, batch_size=128, verbose=1)

# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(test_data, test_labels, batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

# dump he network architecture and weights to file
print("[INFO] dumping architecture and weights to file...")
model.save(model_path)

t_end = datetime.now()
t_diff = t_end - t_start
print("[INFO] total time: ", t_diff)