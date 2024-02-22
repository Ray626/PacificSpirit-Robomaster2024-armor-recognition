import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
#call camera
# cap = cv2.VideoCapture(0)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
# cap.set(10,150)


def getContour(img):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 0, 155])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(imgHsv, lower, upper)
    cv2.imshow("mask",mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = list(contours)
    contours.sort(key= lambda s: cv2.contourArea(s), reverse = True)
    widthArr = []
    lengthArr = []
    pointArr = []

    for conts in contours[:5]:
        area = cv2.contourArea(conts)
        print(area)
        if area > 500:
            x, y, w, h = cv2.boundingRect(conts)
            try:
                if h / w >= 2 and h / frameHeight > 0.01:
                    cv2.drawContours(img, conts, -1, (255, 255, 50), 3)
                    widthArr.append(w)
                    lengthArr.append(h)
                    pointArr.append([x, y])
            except:
                continue

    point = [0,0]

    minval = 99999

    for i in range(len(widthArr)-1):
        for j in range(i + 1, len(lengthArr)-1):
            value = abs(widthArr[i] * lengthArr[i] - widthArr[j] * lengthArr[j])
            print(value)
            if value < minval:
                print("getsome")
                minval = value
                point = [i,j]

    try:
        rect1 = pointArr[point[0]]
        rect2 = pointArr[point[1]]

        point1 = [rect1[0]+widthArr[point[0]]/2,rect1[1]]
        point2 = [rect1[0]+widthArr[point[0]]/2,rect1[1]+lengthArr[point[0]]]
        point3 = [rect2[0]+widthArr[point[1]]/2,rect1[1]]
        point4 = [rect2[0]+widthArr[point[1]]/2,rect1[1]+lengthArr[point[1]]]
        print("I am here")

        print(point1,point2,point3,point4)
        x = np.array([point1,point2,point4,point3],np.int32)
        box = x.reshape((-1,1,2)).astype(np.int32)
        print("I am here")
        cv2.polylines(img,[box],True,(0,255,0),2)
    except:
        pass
    return img


# while True:
#     success, img = cap.read()
#     cv2.imshow("Result", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


cap = cv2.VideoCapture("1234567.mp4")
while True:
    success, img = cap.read()
    img = cv2.resize(img, (frameWidth, frameHeight))
    img= getContour(img)
    cv2.imshow("Result", img)
    if cv2.waitKey(17) & 0xFF == ord('q'):
        break
