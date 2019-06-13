import numpy as n
import cv2 as c


def detect_faces(cascade, test_image, scaleFactor = 1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = c.cvtColor(image_copy, c.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        c.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 7)

    return image_copy



#leemos la imagen, 1 es normal (a color)
img = c.imread("lena.jpg",1)

#con lo que clasificamos para detectar caras.
haar_cascade_face = c.CascadeClassifier('haarcascade_frontalface_default.xml')

#se pasa la img a grises
#se aplica el algoritmos usando nuestro clasificador para detectar caras
#se aplica el algoritmo de Viola-Jones
#se pintan los rectangulos sobre las caras en toda la img
img_facelized = detect_faces(haar_cascade_face,img,1.1)

#motamos la img
c.imshow("image", img_facelized)

#con 0 no dejamos que cierre la ventana
c.waitKey(0)

#guaramos la img con las caras selecionadas
img2 = c.imwrite("face_detected.jpg", img_facelized)

c.destroyAllWindows()
