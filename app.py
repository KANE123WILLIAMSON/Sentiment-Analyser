import tkinter as tk
from tkinter import messagebox
import pickle
import os


# ==========================================
# FILE NAMES
# ==========================================

MODEL_FILE = "emotion_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"


# ==========================================
# CHECK FILES
# ==========================================

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        "emotion_model.pkl not found. "
        "Place it in the same folder as app.py."
    )

if not os.path.exists(VECTORIZER_FILE):
    raise FileNotFoundError(
        "tfidf_vectorizer.pkl not found. "
        "Place it in the same folder as app.py."
    )


# ==========================================
# LOAD MODEL
# ==========================================

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)


# ==========================================
# LOAD TF-IDF VECTORIZER
# ==========================================

with open(VECTORIZER_FILE, "rb") as file:
    vectorizer = pickle.load(file)


# ==========================================
# ANALYZE SENTIMENT
# ==========================================

def analyze_sentiment():

    # Get text from textbox
    text = text_box.get("1.0", tk.END).strip()

    # Check empty text
    if not text:
        messagebox.showwarning(
            "Empty Text",
            "Please enter some text first."
        )
        return

    try:

        # ----------------------------------
        # Convert text into TF-IDF features
        # ----------------------------------

        text_vector = vectorizer.transform([text])

        # ----------------------------------
        # Make prediction
        # ----------------------------------

        prediction = model.predict(text_vector)[0]

        # ----------------------------------
        # Display result
        # ----------------------------------

        result_label.config(
            text=f"Predicted Emotion: {prediction}"
        )

    except Exception as e:

        messagebox.showerror(
            "Prediction Error",
            f"Could not analyze the text.\n\n{e}"
        )


# ==========================================
# CLEAR TEXT
# ==========================================

def clear_text():

    text_box.delete("1.0", tk.END)

    result_label.config(
        text="Predicted Emotion: ---"
    )


# ==========================================
# MAIN WINDOW
# ==========================================

root = tk.Tk()

root.title("Emotion / Sentiment Analysis")

root.geometry("1000x750")

root.configure(
    bg="#f4f6f8"
)


# ==========================================
# TITLE
# ==========================================

title_label = tk.Label(
    root,
    text="Emotion & Sentiment Analysis",
    font=("Arial", 30, "bold"),
    bg="#f4f6f8",
    fg="#222222"
)

title_label.pack(
    pady=(45, 10)
)


# ==========================================
# SUBTITLE
# ==========================================

subtitle_label = tk.Label(
    root,
    text="Enter a sentence and let the model predict its emotion",
    font=("Arial", 14),
    bg="#f4f6f8",
    fg="#666666"
)

subtitle_label.pack(
    pady=(0, 30)
)


# ==========================================
# TEXT BOX
# ==========================================

text_box = tk.Text(
    root,
    height=8,
    width=80,
    font=("Arial", 15),
    wrap=tk.WORD,
    bd=2,
    relief=tk.GROOVE
)

text_box.pack(
    pady=10
)


# ==========================================
# BUTTON FRAME
# ==========================================

button_frame = tk.Frame(
    root,
    bg="#f4f6f8"
)

button_frame.pack(
    pady=25
)


# ==========================================
# ANALYZE BUTTON
# ==========================================

analyze_button = tk.Button(
    button_frame,
    text="Analyze Sentiment",
    command=analyze_sentiment,
    font=("Arial", 13, "bold"),
    bg="#4CAF50",
    fg="white",
    padx=30,
    pady=12,
    cursor="hand2"
)

analyze_button.grid(
    row=0,
    column=0,
    padx=10
)


# ==========================================
# CLEAR BUTTON
# ==========================================

clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_text,
    font=("Arial", 13),
    bg="#e0e0e0",
    fg="#222222",
    padx=30,
    pady=12,
    cursor="hand2"
)

clear_button.grid(
    row=0,
    column=1,
    padx=10
)


# ==========================================
# RESULT
# ==========================================

result_label = tk.Label(
    root,
    text="Predicted Emotion: ---",
    font=("Arial", 22, "bold"),
    bg="#f4f6f8",
    fg="#333333"
)

result_label.pack(
    pady=35
)


# ==========================================
# FOOTER
# ==========================================

footer = tk.Label(
    root,
    text="Powered by Machine Learning",
    font=("Arial", 10),
    bg="#f4f6f8",
    fg="#888888"
)

footer.pack(
    side=tk.BOTTOM,
    pady=20
)


# ==========================================
# START APPLICATION
# ==========================================

root.mainloop()