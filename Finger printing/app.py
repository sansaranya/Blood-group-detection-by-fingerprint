from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
from skimage.feature import hog
import os

app = Flask(__name__)

# Load trained files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

size = 128

# Prediction function
def predict_blood_group(image_path):
    img = cv2.imread(image_path, 0)

    if img is None:
        return "Invalid Image"

    img = cv2.resize(img, (size, size))
    img = cv2.GaussianBlur(img, (5,5), 0)
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


# 🔹 Home Page
@app.route("/")
def index():
    return render_template("home.html")


# 🔹 Prediction Page
@app.route("/predict", methods=["GET", "POST"])
def home():
    result = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            upload_folder = "static"

            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            from werkzeug.utils import secure_filename
            filename = secure_filename(file.filename)

            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)

            result = predict_blood_group(filepath)
            image_path = filepath

    return render_template("index.html", prediction=result, image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)