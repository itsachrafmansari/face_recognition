import cv2
import face_recognition

me = cv2.imread(r'PATH TO YOUR IMAGE')
cvtMe = cv2.cvtColor(me, cv2.COLOR_BGR2RGB)
myFaceLoc = face_recognition.face_locations(cvtMe)
myFaceEnc = face_recognition.face_encodings(cvtMe, myFaceLoc)

camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    cvtFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frameFaceLocs = face_recognition.face_locations(cvtFrame)
    frameFaceEncs = face_recognition.face_encodings(cvtFrame, frameFaceLocs)

    for face, location in zip(frameFaceEncs, frameFaceLocs):
        matches = face_recognition.compare_faces([face], myFaceEnc[0])
        ya, xb, yb, xa = location
        for test in matches:
            if test:
                cv2.rectangle(frame, (xa, ya), (xb, yb), (0, 255, 0), 2)
                print("It's me !")
            else:
                cv2.rectangle(frame, (xa, ya), (xb, yb), (0, 0, 255), 2)
                print("Who is this guy !?")

    cv2.imshow("", frame)
    cv2.waitKey(1)
