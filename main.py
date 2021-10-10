import cv2
import numpy as np


def nothing(x):
    pass


cap = cv2.VideoCapture(0)

cv2.namedWindow("trackbars")
cv2.createTrackbar('th', 'trackbars', 50, 255, nothing)  # borrão
cv2.createTrackbar('erosao', 'trackbars', 1, 255, nothing)  # erosão
cv2.createTrackbar('dil', 'trackbars', 1, 255, nothing)  # dilatação
cv2.createTrackbar('opening', 'trackbars', 1, 255, nothing)  # open
cv2.createTrackbar('closing', 'trackbars', 1, 255, nothing)  # close

while True:
    ret, frame = cap.read()
    #   cv2.imshow('frame', frame)

    #   escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #   cv2.imshow('gray', gray)
    #   blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # THRESHOLD ou borrado
    th = cv2.getTrackbarPos('th', 'trackbars')
    ret, thresh = cv2.threshold(blur, th, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh', thresh)

    # Erosão
    kernel = np.ones((2, 2), np.uint8)
    erosao = cv2.getTrackbarPos('erosao', 'trackbars')
    erosion = cv2.erode(thresh, kernel, iterations=erosao)
    cv2.imshow('erosion', erosion)

    #   Dilatação
    dil = cv2.getTrackbarPos('dil', 'trackbars')
    dilata = cv2.dilate(thresh, kernel, iterations=dil)
    cv2.imshow('dilata', dilata)

    #   open
    op = cv2.getTrackbarPos('opening', 'trackbars')
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=op)
    cv2.imshow('opening', opening)

    #   open
    cl = cv2.getTrackbarPos('closing', 'trackbars')
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=cl)
    cv2.imshow('closing', closing)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
