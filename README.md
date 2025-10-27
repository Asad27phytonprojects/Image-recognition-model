🧠 Offline Handwritten Digit Recognizer

A lightweight handwritten digit classification app built with Python, using a Random Forest model trained on the built-in sklearn digits dataset — runs completely offline, no TensorFlow or internet connection required.

✨ Features

🚀 Offline model — uses load_digits() (8×8 grayscale dataset)

🧩 Random Forest classifier for solid accuracy

🖼️ Drag-and-drop support — drop any image in the console for instant prediction

🔄 Smart inversion detection — automatically checks both normal & inverted versions for better accuracy

📊 Matplotlib visualization — displays predictions with the digit image

💻 Lightweight — no GPU or heavy dependencies needed

🧰 Requirements

Install dependencies using pip:

pip install numpy scikit-learn pillow matplotlib

▶️ How to Run

Clone the repo or download the .py file.

Run:

python project.py


Drag and drop any digit image (e.g., 1.png) into the console window.

See the predicted digit appear along with a visualization.

📸 Example Output
🖼 3.png → Predicted Digit: 3 (Normal version)
Accuracy: 97.85%

🧑‍💻 Tech Stack

Python

scikit-learn

Pillow

Matplotlib

NumPy

💡 Future Improvements

Add batch prediction for multiple images

GUI using tkinter or PyQt

Support for larger 28×28 MNIST dataset
