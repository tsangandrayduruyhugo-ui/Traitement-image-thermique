import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Charger image thermique
image = cv2.imread("image thermique.png")

if image is None:
    print("Erreur : image non trouvée")
    exit()

# 2. Convertir en niveau de gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Réduction du bruit
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 4. Calcul des valeurs thermiques
temp_max = np.max(blur)
temp_min = np.min(blur)
temp_moy = np.mean(blur)

print("Température maximale :", temp_max)
print("Température minimale :", temp_min)
print("Température moyenne :", temp_moy)

# 5. Extraction du profil thermique (ligne au centre)
hauteur, largeur = blur.shape
ligne = hauteur // 2

profil = blur[ligne, :]

# 6. Détection des zones chaudes
seuil_valeur = 200
_, zones_chaudes = cv2.threshold(blur, seuil_valeur, 255, cv2.THRESH_BINARY)

# 7. Trouver la position du point le plus chaud
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blur)

print("Point le plus chaud :", maxLoc)

# Dessiner le point chaud
cv2.circle(image, maxLoc, 10, (0, 0, 255), 2)

# 8. Affichage des images
cv2.imshow("Image thermique", image)
cv2.imshow("Image niveau de gris", gray)
cv2.imshow("Zones chaudes", zones_chaudes)

# 9. Graphique du profil thermique
plt.plot(profil)
plt.title("Profil thermique")
plt.xlabel("Position sur la ligne")
plt.ylabel("Intensité thermique")
plt.grid()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
