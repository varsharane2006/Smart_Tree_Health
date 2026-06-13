from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# LOAD MODEL + CLASSES
# =========================
model = load_model("tree_health_model.h5")

with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

latest_suggestion = ""

# =========================
# PREDICTION FUNCTION
# =========================
def detect(path):

    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    # IMPORTANT: DO NOT rescale here (model already handles it)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    predicted_index = int(np.argmax(prediction[0]))
    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(np.max(prediction)) * 100


    suggestions = {
    "Bacterial_Spot": (
        "🍃 Bacterial Spot",
        "• Remove infected leaves\n"
        "• Avoid excess watering on leaves\n"
        "• Monitor tree regularly"
    ),

    "healthy": (
        "🌿 Healthy Tree",
        "• Continue proper watering\n"
        "• Provide enough sunlight\n"
        "• Keep regular monitoring"
    ),

    "Leaf_Spot": (
        "🍂 Leaf Spot",
        "• Remove affected leaves\n"
        "• Improve air circulation\n"
        "• Apply suitable treatment"
    ),

    "Rot_Disease": (
        "🌱 Rot Disease",
        "• Improve soil drainage\n"
        "• Avoid overwatering\n"
        "• Check root condition"
    ),

    "Rust": (
        "🟠 Rust Disease",
        "• Remove infected parts\n"
        "• Use proper disease control\n"
        "• Keep leaves dry"
    )
}

    title, advice = suggestions.get(
        predicted_class,
        ("Unknown Disease", "No suggestion available.")
    )

    return f"{title} ({confidence:.2f}%)", advice


# =========================
# HOME PAGE
# =========================
@app.route("/", methods=["GET", "POST"])
def home():

    global latest_suggestion

    result = None
    suggestion = None
    image_path = None

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename != "":

            os.makedirs(UPLOAD_FOLDER, exist_ok=True)

            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            result, suggestion = detect(image_path)
            latest_suggestion = suggestion

    return render_template(
        "index.html",
        result=result,
        suggestion=suggestion,
        image_path=image_path
    )


# =========================
# SUGGESTIONS PAGE
# =========================
@app.route("/suggestions")
def suggestions():
    return render_template(
        "suggestions.html",
        suggestion=latest_suggestion
    )


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)