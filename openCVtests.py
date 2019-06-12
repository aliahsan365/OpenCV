import numpy as n
import cv2 as c

#leemos la imagen, 0 es es cala de grises; 1 es normal (a color)
img = c.imread("lena.jpg",0)

# printa una matriz de pixeles (amplada x altura), donde cada pixel es una tripleta
# indicando el color.
print(img)

c.imshow("image", img)

#con 0 no dejamos que cierre la ventana
c.waitKey(0)

#creamos nuestra imagen  a escala de grises
img2 = c.imwrite("escala.jpg", img)

print(img2)

c.destroyAllWindows()
