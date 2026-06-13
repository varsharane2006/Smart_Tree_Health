import tensorflow as tf
import json

# =========================
# LOAD DATASET
# =========================
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=(224, 224),
    batch_size=32,
    seed=123
)

# =========================
# SAVE CLASS NAMES
# =========================
class_names = dataset.class_names
print("Class Names:", class_names)

with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# =========================
# PREFETCH FOR SPEED
# =========================
dataset = dataset.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)

# =========================
# MODEL
# =========================
model = tf.keras.Sequential([

    tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),

    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(256, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

# =========================
# COMPILE
# =========================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# TRAIN
# =========================
model.fit(dataset, epochs=25)

# =========================
# SAVE MODEL
# =========================
model.save("tree_health_model.h5")

print("Training completed successfully!")