import cv2
import face_recognition
from os import walk

multiFacesImage = cv2.imread("PATH TO THE MAGE WHERE FACES WILL BE RECONIZED")
cvtMultiFacesImage = cv2.cvtColor(multiFacesImage, cv2.COLOR_BGR2RGB)
multiFacesLocation = face_recognition.face_locations(cvtMultiFacesImage)
multiFacesEncoding = face_recognition.face_encodings(cvtMultiFacesImage, multiFacesLocation)

recognizedFaceLocations, recognizedFaceEncodings, recognizedFaceNames = [], [], []

for (dirpath, dirnames, filenames) in walk('PATH TO THE IMAGES OF THE RECONIZED FACES'):
    for filename in filenames:
        filepath = "{}/{}".format(dirpath, filename)
        oneFaceImage = cv2.imread(filepath)
        cvtImage = cv2.cvtColor(oneFaceImage, cv2.COLOR_BGR2RGB)
        FaceLocation = face_recognition.face_locations(cvtImage)
        FaceEncoding = face_recognition.face_encodings(cvtImage, FaceLocation)

        recognizedFaceLocations.append(FaceLocation[0])
        recognizedFaceEncodings.append(FaceEncoding[0])
        recognizedFaceNames.append(filename[:-4])


for recognizedface, recognizedlocation in zip(multiFacesEncoding, multiFacesLocation):
    for name, face in zip(recognizedFaceNames, recognizedFaceEncodings):

        confidence = 90  # Percent
        matches = face_recognition.compare_faces([face], recognizedface, tolerance=1 - (confidence / 100))

        ya, xb, yb, xa = recognizedlocation

        recognized = False
        for test in matches:
            if test:
                cv2.rectangle(multiFacesImage, (xa, ya), (xb, yb), (0, 255, 0), 2)
                cv2.rectangle(multiFacesImage, (xa, yb + 30), (xb, yb), (0, 255, 0), cv2.FILLED)
                cv2.putText(multiFacesImage, name, (xa, yb + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
                recognized = True
                break
        if recognized:
            break

cv2.imshow("Done !", multiFacesImage)
cv2.waitKey(5000)
