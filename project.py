import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import random
import os

# Load the offline digits dataset (8x8 grayscale, values 0–16)
digits = load_digits()
X, y = digits.data, digits.target

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(x_train, y_train)

# Evaluate accuracy
accuracy = accuracy_score(y_test, rf.predict(x_test))
print(f"Model trained successfully! ✅")
print(f"Accuracy on test set: {accuracy:.2%}\n")

# Show a few random test predictions
for idx in random.sample(range(len(x_test)), 3):
    img = x_test[idx].reshape(8, 8)
    pred = rf.predict([x_test[idx]])[0]
    actual = y_test[idx]
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.title(f"Predicted: {pred}, Actual: {actual}")
    plt.axis('off')
    plt.show()

# Function to classify an external image (auto tests normal + inverted)
def classify_image(img_path):
    img = Image.open(img_path).convert('L').resize((8, 8), Image.LANCZOS)
    arr = np.array(img).astype(np.float32)
    arr_scaled = (arr / 255.0) * 16.0  # scale 0–255 → 0–16

    # Normal
    sample_normal = arr_scaled.flatten().reshape(1, -1)
    proba_normal = rf.predict_proba(sample_normal)
    pred_normal = rf.predict(sample_normal)[0]
    conf_normal = np.max(proba_normal)

    # Inverted
    arr_inverted = 16.0 - arr_scaled
    sample_inverted = arr_inverted.flatten().reshape(1, -1)
    proba_inverted = rf.predict_proba(sample_inverted)
    pred_inverted = rf.predict(sample_inverted)[0]
    conf_inverted = np.max(proba_inverted)

    # Pick higher confidence
    if conf_inverted > conf_normal:
        best_pred = pred_inverted
        best_arr = arr_inverted
        version = "Inverted"
    else:
        best_pred = pred_normal
        best_arr = arr_scaled
        version = "Normal"

    plt.imshow(best_arr.reshape(8, 8), cmap='gray', interpolation='nearest')
    plt.title(f"{version} Image | Predicted Digit: {best_pred}")
    plt.axis('off')
    plt.show()

    print(f"\n🖼 {os.path.basename(img_path)} → Predicted Digit: {best_pred} ({version} version)\n")

# --- DRAG & DROP feature ---
print("👉 Drag and drop an image file into this window to classify it.")
print("Or type 'exit' to quit.\n")

while True:
    img_path = input("Drop image path here: ").strip().strip('"')
    if img_path.lower() == "exit":
        print("Exiting program 👋")
        break
    if not os.path.exists(img_path):
        print("⚠️ File not found! Try again.")
        continue
    classify_image(img_path)
