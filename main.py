import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot


cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []
blinkCounter = 0
eyeClosedDuration = 0  # new variable to track duration of eye closure
totalEyeClosedDuration = 0  # new variable to track total duration of eye closure
color = (255, 0, 255)

while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 3, color, cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lenghtVer, _ = detector.findDistance(leftUp, leftDown)
        lenghtHor, _ = detector.findDistance(leftLeft, leftRight)

        cv2.line(img, leftUp, leftDown, (0, 200, 0), 2)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 2)

        ratio = int((lenghtVer / lenghtHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        if ratioAvg < 30 and eyeClosedDuration == 0:
            color = (0, 200, 0)
            eyeClosedDuration = 1
        elif eyeClosedDuration != 0:
            eyeClosedDuration += 1
            if eyeClosedDuration > 40:  # eye closure duration threshold
                totalEyeClosedDuration += eyeClosedDuration
                eyeClosedDuration = 0
                color = (255, 0, 255)

        cvzone.putTextRect(img, f'Total Eye Closed Duration: {totalEyeClosedDuration}', (50, 100),
                           colorR=color)

        if totalEyeClosedDuration > 80:  # total eye closure duration threshold
            # trigger alarm or send notification here
            print("Person may have fallen asleep!")
            totalEyeClosedDuration = 0

        imgPlot = plotY.update(ratioAvg, color)
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("Image", imgStack)
    cv2.waitKey(10)
