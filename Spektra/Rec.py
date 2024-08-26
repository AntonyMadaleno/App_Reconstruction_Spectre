import cv2
import numpy as np

from tkinter import filedialog as fd

# Charger les images
img1 = cv2.imread(fd.askopenfilename(), cv2.IMREAD_GRAYSCALE).astype(float)  # Image de référence
img2 = cv2.imread(fd.askopenfilename(), cv2.IMREAD_GRAYSCALE).astype(float)  # Image à aligner

# Normalisation

img1 = ( img1 * 255.0 / np.max(img1) ).astype(np.uint8)
img2 = ( img2 * 255.0 / np.max(img2) ).astype(np.uint8)

# Vérifier que les images sont chargées correctement
if img1 is None or img2 is None:
    print("Erreur: Impossible de charger les images")
    exit()

# Utiliser SIFT pour détecter les points d'intérêt et calculer les descripteurs
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Utiliser le matcher de FLANN pour associer les descripteurs
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Appliquer le ratio test de Lowe pour filtrer les bonnes correspondances
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Vérifier qu'il y a assez de bonnes correspondances
if len(good_matches) > 10:
    # Extraire les points correspondants
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Trouver la matrice homographique
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Aligner l'image
    height, width = img1.shape
    aligned_img = cv2.warpPerspective(img2, M, (width, height))

    # Afficher les images
    cv2.imshow('Image 1', img1)
    cv2.imshow('Image 2', img2)
    cv2.imshow('Aligned Image', aligned_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Pas assez de correspondances trouvées - {}/10".format(len(good_matches)))
