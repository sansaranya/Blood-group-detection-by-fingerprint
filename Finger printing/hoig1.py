import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import hog

# ---------------- DATASET PATH ----------------
dataset_path = "dataset/dataset_blood_group"

features = []
labels = []

size = 128

# ---------------- LOAD DATASET ----------------
for blood_group in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, blood_group)

    if not os.path.isdir(folder_path):
        continue

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        img = cv2.imread(image_path, 0)

        if img is None:
            continue

        # Resize image
        img = cv2.resize(img,(size,size))

        # Noise reduction
        img = cv2.GaussianBlur(img,(5,5),0)

        # Improve contrast
        img = cv2.equalizeHist(img)

        # --------  FEATURE EXTRACTION --------
        hog_features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            visualize=False
        )

        features.append(hog_features)
        labels.append(blood_group)

# ---------------- CONVERT TO ARRAY ----------------
X = np.array(features)
y = np.array(labels)

print("Total images loaded:", len(X))

if len(X) == 0:
    print("Dataset not loaded.")
    exit()

# ---------------- FEATURE SCALING ----------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ---------------- PCA DIMENSION REDUCTION ----------------
pca = PCA(n_components=100)
X = pca.fit_transform(X)

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ---------------- FIND BEST K ----------------
best_k = 1
best_acc = 0

for k in range(1,20):

    model = KNeighborsClassifier(n_neighbors=k, weights='distance')
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test,pred)

    print("K =",k,"Accuracy =",acc)

    if acc > best_acc:
        best_acc = acc
        best_k = k

print("Best K:",best_k)

# ---------------- FINAL MODEL ----------------
model = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print("Model Accuracy: {:.2f}%".format(accuracy * 100))

import joblib

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

# ---------------- PREDICTION FUNCTION ----------------
def predict_blood_group(image_path):

    img = cv2.imread(image_path,0)

    if img is None:
        print("Test image not found.")
        return None

    img = cv2.resize(img,(size,size))
    img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.equalizeHist(img)

    hog_features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        visualize=False
    )

    hog_features = scaler.transform([hog_features])
    hog_features = pca.transform(hog_features)

    prediction = model.predict(hog_features)

    return prediction[0]

# ---------------- TEST IMAGE ----------------
test_image = r"C:\Finger printing\test\O+.BMP"

result = predict_blood_group(test_image)

print("Predicted Blood Group:", result)