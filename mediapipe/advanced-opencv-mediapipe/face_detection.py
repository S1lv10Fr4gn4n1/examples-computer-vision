import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()

cap = cv2.VideoCapture(0)

pTime = 0
cTime = 0

while cv2.waitKey(1) != 27: # Escape
    has_frame, img = cap.read()
    if not has_frame:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    results = faceDetection.process(imgRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection)
            # print(id, detection)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)