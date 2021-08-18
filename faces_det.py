import cv2
import face_recognition

camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()

    cvtFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frameFaceLocs = face_recognition.face_locations(cvtFrame)

    frameFaceEncs = face_recognition.face_encodings(cvtFrame, frameFaceLocs)

    for face, location in zip(frameFaceEncs, frameFaceLocs):
        ya, xb, yb, xa = location

        cv2.rectangle(frame, (xa, ya), (xb, yb), (0, 255, 0), 2)

    cv2.imshow("", frame)
    cv2.waitKey(1)
