import cv2
import numpy as np

frameWidth = 1440
frameHeight = 1080
#call camera
# cap = cv2.VideoCapture(0)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
# cap.set(10,150)
def pnpSolver(image4Points, isBigArmor):

    if (isBigArmor):
        objectPoints = np.array([[-114.5, 27.5, 0], [-114.5, -27.5, 0], [114.5, -27.5, 0], [114.5, 27.5, 0]],
                                dtype=np.float64)
    else:
        objectPoints = np.array([[-67, 27.5, 0], [-67, -27.5, 0], [67, -27.5, 0], [67, 27.5, 0]], dtype=np.float64)

    distCoeffs = np.array([-0.11043461755427092,0.43006333139306097,0.006798237086267663,-0.0008919229727936152,0])
    intrinsics = np.array([[1809.217156286981, 0, 746.8230110511495], [0, 1816.7322159915377, 566.9696264728983],
                           [0, 0, 1]])
    success, rVec, tVec = cv2.solvePnP(objectPoints, image4Points, intrinsics, distCoeffs)
    return rVec, tVec

def getContour(img):
    location = (0,0)
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 150, 180])
    upper = np.array([140, 255, 255])
    find_number_mask = cv2.inRange(imgHsv,np.array([76,77,74]),np.array([103,131,100]))
    mask = cv2.inRange(imgHsv, lower, upper)
    #img_gray = cv2.cvtColor(imgHsv,cv2.COLOR_BGR2GRAY)
    #mask = cv2.threshold(img_gray,mask,)
    find_number = False

    core = np.array([[0.5,0.5,0.5],[0.5,1,0.5],[0.5,0.5,0.5]])
    #mask = cv2.erode(mask,core,iterations=1)
    core = np.array([[0.8, 0.8, 0.8], [0.8, 1, 0.8], [0.8, 0.8, 0.8]])
    mask = cv2.dilate(mask,core,iterations=1)

    cv2.imshow("mask",mask)
    cv2.imshow("num",find_number_mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = list(contours)
    contours.sort(key= lambda s: cv2.contourArea(s), reverse = True)

    contours2, hierarchy2 = cv2.findContours(find_number_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours2 = list(contours2)
    contours2.sort(key=lambda s: cv2.contourArea(s), reverse=True)

    widthArr = []
    lengthArr = []
    pointArr = []

    for conts in contours[:5]:
        area = cv2.contourArea(conts)
        #print(area)
        if area > 280:
           find_number = True

    for conts in contours[:5]:
        area = cv2.contourArea(conts)
        #print(area)
        if area > 280:
            x, y, w, h = cv2.boundingRect(conts)
            try:
                if h / w >= 2 and h / frameHeight > 0.01:
                    cv2.drawContours(img, conts, -1, (255, 255, 50), 3)
                    widthArr.append(w)
                    lengthArr.append(h)
                    pointArr.append([x,y])
            except:
                continue

    point = [0,0]

    minval = 99999

    for i in range(len(widthArr)):
        for j in range(i + 1, len(lengthArr)):
            value = abs(widthArr[i] * lengthArr[i] - widthArr[j] * lengthArr[j])
            #print(value)
            if value < minval:
                minval = value
                point = [i,j]
    try:
        rect1 = pointArr[point[0]]
        rect2 = pointArr[point[1]]

        point1 = [rect1[0]+widthArr[point[0]]/2,rect1[1]-40.14]
        point2 = [rect1[0]+widthArr[point[0]]/2,rect1[1]+lengthArr[point[0]]+40.14]
        point3 = [rect2[0]+widthArr[point[1]]/2,rect2[1]-40.14]
        point4 = [rect2[0]+widthArr[point[1]]/2,rect2[1]+lengthArr[point[1]]+40.14]
        if (point1[0] < point4[0] and point1[1] <= point4[1] and find_number):
            point = [(point1[0]+point4[0])/2,(point1[1]+point4[1])/2]
        cv2.circle(img, (int(point1[0]),int(point1[1])), 5, (255, 255, 255),15)
        cv2.circle(img, (int(point2[0]),int(point2[1])), 2, (0, 0, 0),15)
        cv2.circle(img, (int(point3[0]),int(point3[1])), 2, (0, 255, 0),15)
        cv2.circle(img, (int(point4[0]),int(point4[1])), 2, (0, 0, 255),15)
        cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 0, 255), 15)
        # print(point1,point2,point3,point4)
        if point1[0] < point3[0]:
            x = np.array([point1, point2, point4, point3], np.int32)
        else:
            x = np.array([point4,point3,point1,point2],np.int32)

        box = x.reshape((-1,1,2)).astype(np.int32)
        print(box)

        pnpIn = np.array([box[0][0], box[1][0], box[2][0], box[3][0]], dtype=np.float64)

        rvec, tvec = pnpSolver(pnpIn, False)
        print(tvec)
        cv2.polylines(img,[box],True,(0,255,0),2)
        location = point
    except:
        pass
    return img, location


# while True:
#     success, img = cap.read()
#     cv2.imshow("Result", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

cap = cv2.VideoCapture("test sample/blue_shield.mp4")
while True:
    success, img = cap.read()
    try:
        img = cv2.resize(img, (frameWidth, frameHeight))
        img, location = getContour(img)
        cv2.imshow("Result", img)
        if cv2.waitKey(36) & 0xFF == ord('q'):
            break
    except:
        break
