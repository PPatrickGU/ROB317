import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import shape
from skimage.feature import hog

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
    plt.axhline(y=0.92, ls="-", c="red")
    plt.ylim([0.4, 1])
    plt.title("Correlation des histogrammes h et h-1")
    plt.xlabel("Numero de frames")
    plt.ylabel("Correlation (%)")
    plt.draw()
    plt.savefig('Q4_correlation_hog_v3.jpg')
    plt.pause(0.0001)

def calculate_hog(image, number_ori):
    fd, hog_image = hog(image, orientations=number_ori, pixels_per_cell=(16, 16),
						cells_per_block=(1, 1), visualize=True)#, multichannel=True)
    #histogram_hog = normalize(hog_image)

    number_histograms = (int(shape(image)[0]/16)*int(shape(image)[1]/16))
    test = np.reshape(np.array(fd), (number_histograms, number_ori))
    histogram_hog = np.sum(test, axis=0)
    histogram_hog = np.reshape(histogram_hog, (number_ori))
    histogram_hog[:] = (histogram_hog[:]/np.max(histogram_hog))*256

    return histogram_hog

#Ouverture du flux video
cap = cv2.VideoCapture("TP2_Videos/Extrait3-Vertigo-Dream_Scene(320p).m4v")
#cap = cv2.VideoCapture(0)

ret, frame = cap.read() # Passe à l'image suivante
yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

hist_uv = cv2.calcHist([yuv_frame], [1,2], None, [256,256],[0,256,0,256])
hist_hog = calculate_hog(gray_frame, 8)

hist_uv_prev = hist_uv
hist_hog_prev = hist_hog
corr_uv = []
corr_hog = []

cv2.namedWindow('Histogramme', cv2.WINDOW_NORMAL)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
index = 1

ret, frame = cap.read() # Passe à l'image suivante
yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while(ret):
    index += 1

    #hist_uv = cv2.calcHist([yuv_frame], [1, 2], None, [256, 256], [0, 256, 0, 256])
    hist_hog = calculate_hog(gray_frame, 8)

    corr_uv.append(cv2.compareHist(hist_uv_prev, hist_uv, 0))
    #corr_hog.append(cv2.compareHist(hist_hog_prev, hist_hog, 0))
    corr_hog.append(np.corrcoef(hist_hog_prev, hist_hog)[0, 1])

    cv2.imshow('Histogramme', hist_uv)
    cv2.imshow('Video', frame)
    ## Plot en temps réel de la valeur de la correlation
    plot_correlation(corr_hog)

    hist_uv_prev = hist_uv
    hist_hog_prev = hist_hog

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
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

cap.release()
cv2.destroyAllWindows()