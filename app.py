import fnmatch
import os
import streamlit as st
import spacy
import pickle
from main import read_results

@st.cache_resource
def load_model_and_vectorizer(a_vectorizer_file_name, a_classifier_file_name):
    with open(a_vectorizer_file_name, "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    with open(a_classifier_file_name, "rb") as model_file:
        classifier = pickle.load(model_file)
    return vectorizer, classifier

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

def file_selector(folder_path='.', a_fnmatch='*', a_label='Select a file'):
    filenames = os.listdir(folder_path)
    filtered_filenames = []
    for fn in filenames:
        if fnmatch.fnmatch(fn, a_fnmatch):
            filtered_filenames.append(fn)
    selected_filename = st.selectbox(a_label, filtered_filenames)
    return os.path.join(folder_path, selected_filename)

def preprocess_text(text, nlp):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def analyze_sentiment(review, vectorizer, classifier, nlp):
    processed_review = preprocess_text(review, nlp)
    vectorized_review = vectorizer.transform([processed_review])
    prediction = classifier.predict(vectorized_review)[0]
    sentiment = "Pozytywny" if prediction == 1 else "Negatywny"
    return sentiment

def main():
    st.markdown(
        "<h2 style='text-align: center; color: #4CAF50;'>🎥 Analiza sentymentu recenzji 🎥</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center;'>Analizuj sentyment swojej recenzji za pomocą modelu różnych klasyfikatorów!</p>",
        unsafe_allow_html=True
    )

    nlp = load_spacy_model()

    vectorizer_name = file_selector("vectorizers/", a_label="Wybierz wektoryzator:")
    st.write('Wybrałeś wektoryzator: `%s`' % vectorizer_name)

    classifier_name = file_selector("models/", '*' + vectorizer_name[12:-4] + '*', "Wybierz klasyfikator:")
    st.write('Wybrałeś klasyfikator: `%s`' % classifier_name)

    review = st.text_area(
        "Wpisz recenzję filmu:",
        height=200,
        placeholder="Napisz swoją recenzję filmu tutaj (np. 'Film był fantastyczny!')"
    )

    if st.button("🔍 Analizuj sentyment"):
        if review.strip():
            vectorizer, classifier = load_model_and_vectorizer(vectorizer_name, classifier_name)
            sentiment = analyze_sentiment(review, vectorizer, classifier, nlp)
            if sentiment == "Pozytywny":
                st.success("🎉 Sentyment recenzji: **Pozytywny** 🎉")
            else:
                st.error("😞 Sentyment recenzji: **Negatywny** 😞")
        else:
            st.warning("Proszę wpisać recenzję do analizy.")

    if "show_examples" not in st.session_state:
        st.session_state.show_examples = False

    if st.button("Pokaż/Ukryj przykładowe recenzje"):
        st.session_state.show_examples = not st.session_state.show_examples

    if st.session_state.show_examples:
        examples = [
            "The movie was absolutely fantastic! The actors were brilliant.",
            "It was a terrible movie. I wouldn't recommend it to anyone.",
            "An average movie with some interesting moments, but overall boring."
        ]
        st.write("Oto przykładowe recenzje, które możesz przetestować:")
        for example in examples:
            st.write(f"- {example}")

    st.markdown(
        "<br/> <p style='text-align: center;'>Ranking klasyfikatorów:</p>",
        unsafe_allow_html=True
    )

    st.dataframe(read_results())

if __name__ == "__main__":
    main()
