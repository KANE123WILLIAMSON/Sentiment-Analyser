import streamlit as st
import pickle
import re


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Emotion & Sentiment Analysis",
    layout="centered"
)


# ==========================================
# LOAD MODEL AND VECTORIZER
# ==========================================

MODEL_FILE = "emotion_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"


@st.cache_resource
def load_model():

    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_FILE, "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


model, vectorizer = load_model()


# ==========================================
# TEXT PREPROCESSING
# ==========================================

def preprocess_text(text):

    # Remove unnecessary spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# ==========================================
# TITLE
# ==========================================

st.title("Emotion & Sentiment Analysis")

st.write(
    "Enter a sentence below and let the machine learning "
    "model identify the emotion expressed in the text."
)


# ==========================================
# TEXT INPUT
# ==========================================

text = st.text_area(
    "Enter your text:",
    placeholder="Example: I am extremely happy today!",
    height=180
)


# ==========================================
# ANALYZE BUTTON
# ==========================================

if st.button(
    "Analyze Sentiment",
    use_container_width=True
):

    if not text.strip():

        st.warning(
            "Please enter some text first."
        )

    else:

        try:

            # ----------------------------------
            # PREPROCESS INPUT
            # ----------------------------------

            cleaned_text = preprocess_text(text)


            # ----------------------------------
            # TF-IDF TRANSFORMATION
            # ----------------------------------

            text_vector = vectorizer.transform(
                [cleaned_text]
            )


            # ----------------------------------
            # PREDICTION
            # ----------------------------------

            prediction = model.predict(
                text_vector
            )[0]


            # ----------------------------------
            # DISPLAY PREDICTION
            # ----------------------------------

            st.success(
                f"### Predicted Emotion: {prediction}"
            )


            # ----------------------------------
            # CONFIDENCE SCORE
            # ----------------------------------

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    text_vector
                )[0]

                confidence = max(probabilities) * 100

                st.write(
                    f"**Model Confidence: {confidence:.2f}%**"
                )

                st.progress(
                    min(confidence / 100, 1.0)
                )


                # ----------------------------------
                # TOP 3 PREDICTIONS
                # ----------------------------------

                if hasattr(model, "classes_"):

                    classes = model.classes_

                    results = list(
                        zip(classes, probabilities)
                    )

                    results.sort(
                        key=lambda x: x[1],
                        reverse=True
                    )

                    st.write("### Top Predictions")

                    for emotion, probability in results[:3]:

                        st.write(
                            f"**{emotion}** — "
                            f"{probability * 100:.2f}%"
                        )


        except Exception as e:

            st.error(
                f"Could not analyze the text: {e}"
            )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Powered by Machine Learning & Natural Language Processing"
)