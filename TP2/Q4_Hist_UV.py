import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import shape

for option in cv2.__dict__:
    if 'CORREL' in option:
        print (option)

def normalize(hist) :
    valmax = np.amax(hist)
    valmin = np.amin(hist)
    hist = hist / (valmax - valmin) * 255
    return hist

def plot_correlation(corr):
    plt.figure(num=1, figsize=(8, 4))
    plt.clf()
    plt.plot(corr, 'b', linewidth = 0.5)
    plt.axhline(y=0.6, ls="-", c="red")
    plt.ylim([0, 1])
    plt.title("Correlation des histogrammes h et h-1")
    plt.xlabel("Numero de frames")
    plt.ylabel("Correlation (%)")
    plt.draw()
    plt.savefig('Q4_correlation_hist_uv_v5.jpg')
    plt.pause(0.0001)


#Ouverture du flux video
cap = cv2.VideoCapture("TP2_Videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")

ret, frame = cap.read() # Passe à l'image suivante
yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
hist_uv = cv2.calcHist([yuv_frame], [1,2], None, [256,256],[0,256,0,256])

hist_uv_prev = hist_uv
corr_uv = []

cv2.namedWindow('Histogramme', cv2.WINDOW_NORMAL)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
index = 1

ret, frame = cap.read() # Passe à l'image suivante
yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

while(ret):
    index += 1

    hist_uv = cv2.calcHist([yuv_frame], [1, 2], None, [256, 256], [0, 256, 0, 256])
    corr_uv.append(cv2.compareHist(hist_uv_prev, hist_uv, 0))

    cv2.imshow('Histogramme', hist_uv)
    cv2.imshow('Video', frame)
    ## Plot en temps réel de la valeur de la correlation
    plot_correlation(corr_uv)

    hist_uv_prev = hist_uv

    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame)
        cv2.imwrite('YUV_Frame_%04d.png' % index, yuv_frame)
    elif k == ord('q'):
        break

    ret, frame = cap.read()
    if (ret):
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

cap.release()
cv2.destroyAllWindows()