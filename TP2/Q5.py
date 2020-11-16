import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import shape
import math

for option in cv2.__dict__:
    if 'CORREL' in option:
        print (option)

def normalize(hist) :
    valmax = np.amax(hist)
    valmin = np.amin(hist)
    hist = hist / (valmax - valmin) * 255
    return hist

def plot_correlation(corr, show = True):
    plt.figure(num=1, figsize=(10, 6))
    plt.clf()
    plt.plot(corr, 'b', linewidth = 0.5)
    plt.ylim([0, 1])
    plt.title("Correlation des histogrammes h et h-1")
    plt.xlabel("Numero de frames")
    plt.ylabel("Correlation (%)")
    if show == True:
        plt.draw()
        plt.pause(0.0001)


def Image_clef_index(cut_index, nombre_plan, nombre_point_anguleux):
    image_clef_index = []
    tmp = nombre_point_anguleux[0:cut_index[0]]
    image_clef_index.append(1 + tmp.index(max(tmp)))
    for i in range(nombre_plan - 2):
        tmp = nombre_point_anguleux[cut_index[i]:cut_index[i + 1]]
        image_clef_index.append(1 + tmp.index(max(tmp)) + cut_index[i])
    tmp = nombre_point_anguleux[cut_index[i+1]:]
    image_clef_index.append(1 + tmp.index(max(tmp)) + cut_index[i+1])
    return image_clef_index

# Paramètres du détecteur de points d'intérêt
feature_params = dict( maxCorners = 10000,
                       qualityLevel = 0.01,
                       minDistance = 5,
                       blockSize = 7 )

# Paramètres de l'algo de Lucas et Kanade
lk_params = dict( winSize  = (15,15),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# Découper le plan
cap = cv2.VideoCapture("TP2_Videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
ret, frame = cap.read() # Passe à l'image suivante
frame1 = frame.copy()
yuv_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)
hist = cv2.calcHist([yuv_frame], [1,2], None, [256,256],[0,256,0,256])
hist_old = hist
corr = []
cut_index = []
index = 1

while(ret):

    frame1 = frame.copy()
    yuv_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)
    hist = cv2.calcHist([yuv_frame], [1, 2], None, [256, 256], [0, 256, 0, 256])
    hist = normalize(hist).astype('float32')
    corr.append(cv2.compareHist(hist_old, hist, 0))

    if corr[index-1] < 0.5*1: #seuil = 0.5
        if index-1 not in cut_index:
            cut_index.append(index)
    # cv2.imshow('Histogramme', hist)
    # cv2.imshow('Video', frame)
    plot_correlation(corr, False)
    hist_old = hist

    ret, frame = cap.read()
    index += 1

cap.release()
cv2.destroyAllWindows()

# Obtenir l'image clef index
cap = cv2.VideoCapture("TP2_Videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
ret, frame = cap.read() # Passe à l'image suivante

frame2 = frame.copy()
old_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
nombre_point_anguleux = []

while(ret):

    nombre_point_anguleux.append(p0.shape[0])
    frame2 = frame.copy()
    frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)     # Calcul du flot optique

    # Sélection des points valides
    good_new = p1[st==1]
    good_old = p0[st==1]

    # Image masque pour tracer les vecteurs de flot
    mask = np.zeros_like(frame)
    # Affichage des vecteurs de flot
    # for i,(new,old) in enumerate(zip(good_new,good_old)):
    #     a,b = new.ravel()
    #     c,d = old.ravel()
    #     mask = cv2.line(mask, (a,b),(c,d),(255,255,0),2)
    #     frame = cv2.circle(frame,(c,d),3,(255,255,0),-1)
    # img = cv2.add(frame2,mask)
    # cv2.imshow('Flot Optique Lucas-Kanade Pyramidal',img)
    # Mis à jour image et détection des nouveaux points
    p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
    old_gray = frame_gray.copy()

    ret, frame = cap.read()
    index += 1

cap.release()
cv2.destroyAllWindows()

print("Index decoupage: ", cut_index)

nombre_plan = len(cut_index) + 1 # nombre des plans
image_clef_index = []
print("nombre de plan: ", nombre_plan)

image_clef_index = Image_clef_index(cut_index, nombre_plan, nombre_point_anguleux)

print("Index image-celf: ", image_clef_index)

# Sauvegarder l'image clef
cap = cv2.VideoCapture("TP2_Videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
ret, frame = cap.read() # Passe à l'image suivante
index = 1
plan_index = 0

while(ret and plan_index < nombre_plan):
    if index == image_clef_index[plan_index]:
        cv2.imwrite('%d.png' %index, frame)
        plan_index += 1
    ret, frame = cap.read()
    index += 1

print('Toutes les images-clefs sont sauvegardées !!!')
