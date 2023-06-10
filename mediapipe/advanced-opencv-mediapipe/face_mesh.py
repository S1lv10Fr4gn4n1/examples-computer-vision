import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

pTime = 0
cTime = 0

while cv2.waitKey(1) != 27:  # Escape
    has_frame, img = cap.read()
    if not has_frame:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for landmark in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, landmark, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
