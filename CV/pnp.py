import cv2
import numpy as np


def pnpSolver(image4Points, isBigArmor):

    if (isBigArmor):
        objectPoints = np.array([[-114.5, 27.5, 0], [114.5, 27.5, 0], [114.5, -27.5, 0], [-114.5, -27.5, 0]],
                                dtype=np.float64)
    else:
        objectPoints = np.array([[-67, 27.5, 0], [67, 27.5, 0], [67, -27.5, 0], [-67, -27.5, 0]], dtype=np.float64)

    distCoeffs = np.array([-0.11043461755427092,0.43006333139306097,0.006798237086267663,-0.0008919229727936152,0])
    intrinsics = np.array([[1809.217156286981, 0, 746.8230110511495], [0, 1816.7322159915377, 566.9696264728983],
                           [0, 0, 1]])
    success, rVec, tVec = cv2.solvePnP(objectPoints, image4Points, intrinsics, distCoeffs)
    return rVec, tVec


pnpIn = np.array([[491,113], [491,274], [171,268], [171,107]], dtype=np.float64)

rvec, tvec = pnpSolver(pnpIn, False)
print(tvec)