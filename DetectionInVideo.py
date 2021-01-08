
import numpy as np
import argparse
import imutils
import cv2
import time
import os

# path za input i autput
videopath = "AnkaraFoodMarket_Trim2.mp4"
outputpath = "output.avi"

ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", default='yolo-coco')
ap.add_argument("-c", "--confidence", type=float, default=0.5)
ap.add_argument("-t", "--threshold", type=float, default=0.3)
args = vars(ap.parse_args())

#path do yolo filova
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
# generira random boje za labele, pronasao ovaj kod na githubu u nekom projektu, radi odlicno
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# pre trained yolo weights i configuration path
weightPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])


# Deep-Neural Network model od OpenCV2 cita skinute weightove
net = cv2.dnn.readNetFromDarknet(configPath, weightPath)


# vadim van samo output layerse
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# DIO KODA S VIDEOM

#iniciram video stream pomocu cv2
vs = cv2.VideoCapture(videopath)
writer = None
(W, H) = (None, None)

# odredjivanje broja framova u videu
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("Broj framove u videu {}".format(total))
except:
    print("nije moguce odrediti broj frejmova u videu")
    total = -1;


# prikaz broja sekundi
seconds = 1

#citamo cijeli video
while True:
    # citamo sljedeci frame iz input fila
    (grabbed, frame) = vs.read()
    # izlazimo iz whila ako smo dosli do kraja videa
    if not grabbed:
        break;
    # uzimanje dimenzija
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # blob - 4D numpy array, podsjetnik za mene
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)

    #vrijeme gledam ovdje koliko treba da se izvede
    start = time.time()
    # dobivamo autput s kojim radim dalje
    layersOutputs = net.forward(ln)
    end = time.time()

    # za svaki frame ovo radimo
    boxes = []
    confidences = []
    classIDs = []

   #prijelaz preko svakog output layera
    for output in layersOutputs:
        # prijelaz preko svakog detektiranog objekta u outuputu
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > args["confidence"]:
                # za crtanje oko pronadjenog objekta pomocu boxa
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                #sluzit ce za ispit iznad boxa
                confidences.append(float(confidence))
                classIDs.append(classID)

        #ovo radi...
        # applying the non max suppresion to get rid of duplicate detections based on fixed thresholds
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

       #barem jedan objekt mora biti biti detektiran za dalje
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # boje
                color = [int(c) for c in COLORS[classIDs[i]]]
                # rectangles za iscrtavane
                # (you can adjust the last parameter to increase the boldness of the box)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 6)
                # labela za confidence score
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                # you can change the text properties below
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 4)

        # provjera za writer, skint s githuba
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(outputpath, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
            if total > 0:
                elap = (end - start)
                print("single frame took {} seconds".format(elap))
                print("estimated total time to finish: {}".format(elap * total))
        writer.write(frame)
    print("Seconds: {}".format(seconds))
    seconds = seconds + 1
# ako video je krivi ili ne podrzan format ili path javiti e gresku ovdje u kodu
writer.release()
vs.release()
