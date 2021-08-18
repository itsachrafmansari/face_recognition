import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    _, frame = cap.read()

    frameload = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frameload, 1.1, 4, minSize=(100, 100))

    for (a, b, c, d) in faces:
        cv2.rectangle(frame, (a, b), (a + c, b + d), (255, 0, 0), 2)

    cv2.imshow("bl", frame)
    cv2.waitKey(1)
