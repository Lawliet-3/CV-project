import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
#cap.set(cv2.CAP_PROP_FPS, 60)

segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

listImg = os.listdir("Images")

imgList = []
for imgPath in listImg:
    img = cv2.imread(f'Images/{imgPath}')
    imgList.append(img)

IndexImg = 0

while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imgList[IndexImg], threshold=0.4)

    imgStacked = cvzone.stackImages([img,imgOut], 2, 1)
    _, imgStacked = fpsReader.update(imgStacked, color=(0,0,255))
    print(IndexImg)
    cv2.imshow("Image", imgStacked)
    key = cv2.waitKey(1)
    if key == ord('a'):
        if IndexImg > 0:
            IndexImg -= 1
    elif key == ord('d'):
        if IndexImg < len(imgList) - 1:
            IndexImg += 1
    elif key == ord('q'):
        break
cap.release()
cv2.destroyAllWindow()
