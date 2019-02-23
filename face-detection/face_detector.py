import cv2
import numpy as np

class FaceDetector(object):
    
    def __init__(self, prototxt, model, min_confidence):
        # load our serialized model from disk
        self.min_confidence = min_confidence
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

    def detect(self, image):
        # load the input image and construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        (h, w, _) = image.shape

        normalized_size = (300, 300)
        resized_image = cv2.resize(image, normalized_size)
        blob = cv2.dnn.blobFromImage(resized_image, 1.0, normalized_size, (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the predition
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the 'confidence' is greater then the minimum confidence
            if confidence > self.min_confidence:
                # compute the (x,y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w,h,w,h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the bounding box of the face along with the associated probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        return image