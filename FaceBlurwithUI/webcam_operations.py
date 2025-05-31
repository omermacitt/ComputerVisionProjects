import cv2 as cv
from cv2.data import haarcascades


def blur_image(img, x, y, w, h,blur):
    blur_image = cv.GaussianBlur(img, (blur, blur), 0)
    blur_image[y:y + h,x:x + w] = img[y:y + h,x:x + w]
    return blur_image


def detect_face(img, edge, blur):
    blur = int(blur)
    if blur < 1:
        blur = 1
    if blur % 2 == 0:
        blur += 1

    edge = int(edge)
    if edge < 0:
        edge = 0
    haarcascade = cv.CascadeClassifier(haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray, 1.3, 4)
    h_img, w_img = img.shape[:2]
    for (x, y, w, h) in faces:
        # 3) Koordinatları resim sınırlarına göre kırp
        x1 = max(x - edge, 0)
        y1 = max(y - edge, 0)
        x2 = min(x + w + edge, w_img)
        y2 = min(y + h + edge, h_img)

        # 4) Blur işlemini uygula
        img = blur_image(img, x1, y1, x2 - x1, y2 - y1, blur)
    return img

def start_webcam():
    capture = cv.VideoCapture("https://192.168.1.8:8080/video")
    while True:
        success, img = capture.read()
        # img = detect_face(img)
        cv.imshow("Video", img)
        if cv.waitKey(20) & 0xFF == ord("d"):
            break
    capture.release()
    cv.destroyAllWindows()