import cv2 as cv
from cv2.data import haarcascades


def blur_image(img, x, y, w, h):
    blur_image = cv.GaussianBlur(img, (15, 15), 0)
    blur_image[y:y + h,x:x + w] = img[y:y + h,x:x + w]
    return blur_image


def detect_face(img):
    haarcascade = cv.CascadeClassifier(haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        img = blur_image(img, x - 20, y - 20, w + 20, h + 20)
    return img

capture = cv.VideoCapture("https://192.168.1.8:8080/video")
while True:
    success, img = capture.read()
    img = detect_face(img)
    cv.imshow("Video", img)
    if cv.waitKey(20) & 0xFF == ord("d"):
        break
capture.release()
cv.destroyAllWindows()