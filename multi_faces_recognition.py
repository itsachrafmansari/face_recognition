import cv2
import face_recognition
from pathlib import Path

# A directory that contains at least one image of a known faces with its name (image's name) as the person's name
dir2knownFaces = r"DIRECTORY/OF/KNOWN/FACES/"
# The image where faces will get extracted and compared to the known faces
targetedImage = r"PATH/TO/TARGETED/IMAGE"
# Level of confidence needed to consider two faces as matched (min : 0, max : 100)
confidence = 90

def imageEncoding(path2image):
    imgName = str(Path(path2image).stem)
    loadImg = cv2.imread(str(path2image))
    cvtImg = cv2.cvtColor(loadImg, cv2.COLOR_BGR2RGB)
    faceLocations = face_recognition.face_locations(cvtImg)
    faceEncodings = face_recognition.face_encodings(cvtImg, faceLocations)
    return loadImg, imgName, faceEncodings, faceLocations


# A set of images, each image contains one face and its name represent the person's name
knownFaceEncodings, knownFaceNames = [], []

files = [x for x in Path(dir2knownFaces).iterdir() if x.is_file()]
for imgPath in files:
    filename = Path(imgPath).stem
    _, faceNames, faceEncs, _ = imageEncoding(imgPath)
    knownFaceNames.append(faceNames)
    knownFaceEncodings.append(faceEncs[0])


# An image to extract and recognize faces from
trgtImgLoad, _, trgtFaceEncs, trgtFaceLocs = imageEncoding(targetedImage)

for face, location in zip(trgtFaceEncs, trgtFaceLocs):
    for knownFaceName, knownFaceEncoding in zip(knownFaceNames, knownFaceEncodings):

        matche = face_recognition.compare_faces([knownFaceEncoding], face, tolerance=1 - (confidence / 100))
        ya, xb, yb, xa = location
        recognized = False

        for test in matche:
            if test:
                cv2.rectangle(trgtImgLoad, (xa, ya), (xb, yb), (0, 255, 0), 2)
                cv2.rectangle(trgtImgLoad, (xa, yb + 30), (xb, yb), (0, 255, 0), cv2.FILLED)
                cv2.putText(trgtImgLoad, knownFaceName, (xa, yb + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
                recognized = True
                break

        if recognized:
            break

cv2.imshow("Done !", trgtImgLoad)
cv2.waitKey(5000)
