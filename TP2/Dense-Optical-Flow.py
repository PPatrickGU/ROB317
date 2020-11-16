import cv2
import numpy as np

def normalize(hist) :
    valmax = np.amax(hist)
    valmin = np.amin(hist)
    hist = hist / (valmax - valmin) * 255
    return hist

def histgram2d(frame, a, b):
    hist = cv2.calcHist([frame], [1, 0], None, [512,512], [-64, 64, -64, 64])
    # hist = cv2.calcHist([frame], [1, 0], None, [500, 500], [0, 50, 0, 50])
    hist= cv2.cvtColor(normalize(hist).astype('float32'), cv2.COLOR_GRAY2BGR)
    return hist

cv2.namedWindow('Histogramme', cv2.WINDOW_NORMAL)
cv2.namedWindow('Image et Champ de vitesses (Farneback)', cv2.WINDOW_NORMAL)

#Ouverture du flux video
cap = cv2.VideoCapture("E:/ROB317/TP2/TP2_Videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
# cap = cv2.VideoCapture(0)
ret, frame1 = cap.read() # Passe à l'image suivante
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris
hsv = np.zeros_like(frame1) # Image nulle de même taille que frame1 (affichage OF)
hsv[:,:,1] = 255 # Toutes les couleurs sont saturées au maximum

index = 1
ret, frame2 = cap.read()
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 


while(ret):
    index += 1
    flow = cv2.calcOpticalFlowFarneback(prvs,next,None, 
                                        pyr_scale = 0.5,# Taux de réduction pyramidal
                                        levels = 3, # Nombre de niveaux de la pyramide
                                        winsize = 15, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations = 3, # Nb d'itérations par niveau
                                        poly_n = 7, # Taille voisinage pour approximation polynomiale
                                        poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées 
                                        flags = 0)	
    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1]) # Conversion cartésien vers polaire
    hsv[:,:,0] = (ang*180)/(2*np.pi) # Teinte (codée sur [0..179] dans OpenCV) <--> Argument
    hsv[:,:,2] = (mag*255)/np.amax(mag) # Valeur <--> Norme 

    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    result = np.vstack((frame2, bgr))


    cv2.imshow('Image et Champ de vitesses (Farneback)', result.astype(np.uint8))

    hist1 = histgram2d(flow, 1, 0)
    cv2.imshow('Histogramme', hist1)

    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        # cv2.imwrite('Frame_%04d.png'%index,frame2)
        # cv2.imwrite('OF_hsv_%04d.png'%index,bgr)
        cv2.imwrite('hist_t_x.png',  hist1 * 255)
    prvs = next
    ret, frame2 = cap.read()
    if (ret):
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

cap.release()
cv2.destroyAllWindows()
