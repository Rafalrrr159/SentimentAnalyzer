import fnmatch
import os
import streamlit as st
import spacy
import pickle
from main import read_results
import pandas as pd
import random

@st.cache_resource
def load_model_and_vectorizer(a_vectorizer_file_name, a_classifier_file_name):
    with open(a_vectorizer_file_name, "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    with open(a_classifier_file_name, "rb") as model_file:
        classifier = pickle.load(model_file)
    return vectorizer, classifier

@st.cache_resource
def load_best_models_from_results():
    if not os.path.exists("results/all_results.csv"):
        st.error("Plik all_results.csv nie został znaleziony. Uruchom najpierw skrypt treningowy.")
        return []

    results_df = pd.read_csv("results/all_results.csv")
    best_models = []

    combinations = results_df.groupby(["Vectorizer", "Classifier"])

    for (vectorizer, classifier), group in combinations:
        best_accuracy = group.loc[group["Test Accuracy"].idxmax()]
        best_precision = group.loc[group["Test Precision"].idxmax()]
        best_recall = group.loc[group["Test Recall"].idxmax()]

        for best_model, metric in zip([best_accuracy, best_precision, best_recall], ["Test Accuracy", "Test Precision", "Test Recall"]):
            classifier_model_file = os.path.join("models", best_model["Classifier Model File"])
            vectorizer_model_file = os.path.join("vectorizers", best_model["Vectorizer Model File"])

            if os.path.exists(classifier_model_file) and os.path.exists(vectorizer_model_file):
                with open(classifier_model_file, "rb") as model_f:
                    classifier_model = pickle.load(model_f)
                with open(vectorizer_model_file, "rb") as vec_f:
                    vectorizer_model = pickle.load(vec_f)

                best_models.append({
                    "vectorizer": vectorizer,
                    "classifier": classifier,
                    "vectorizer_model": vectorizer_model,
                    "classifier_model": classifier_model,
                    "metric": metric
                })

    return best_models

@st.cache_resource
def load_test_reviews():
    if not os.path.exists("test_reviews.csv"):
        st.error("Plik test_reviews.csv nie został znaleziony. Upewnij się, że skrypt main.py został uruchomiony.")
        return [], []

    test_data = pd.read_csv("test_reviews.csv")
    return test_data["review"].tolist(), test_data["sentiment"].tolist()


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

def analyze_sentiment_list_of_models(review, models, nlp):
    processed_review = preprocess_text(review, nlp)
    results = []

    for model in models:
        classifier_model = model["classifier_model"]
        vectorizer_model = model["vectorizer_model"]

        vectorized_review = vectorizer_model.transform([processed_review])
        prediction = classifier_model.predict(vectorized_review)[0]

        sentiment = "✔ Pozytywny ✔" if prediction == 1 else "❌ Negatywny ❌"
        results.append({
            "Metoda wektoryzacji": model["vectorizer"],
            "Klasyfikator": model["classifier"],
            "Miara": model["metric"],
            "Sentyment": sentiment
        })

    return results

def main():
    st.markdown(
        "<h2 style='text-align: center; color: #4CAF50;'>🎥 Analiza sentymentu recenzji 🎥</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center;'>Analizuj sentyment swojej recenzji za pomocą modelu różnych klasyfikatorów!</p>",
        unsafe_allow_html=True
    )

    models = load_best_models_from_results()
    if not models:
        st.stop()

    nlp = load_spacy_model()

    vectorizer_name = file_selector("vectorizers/", a_label="Wybierz wektoryzator:")
    st.write('Wybrałeś wektoryzator: `%s`' % vectorizer_name)

    classifier_name = file_selector("models/", '*' + vectorizer_name[12:-4] + '*', "Wybierz klasyfikator:")
    st.write('Wybrałeś klasyfikator: `%s`' % classifier_name)

    test_reviews, test_sentiments = load_test_reviews()

    if "review" not in st.session_state:
        st.session_state.review = ""
        st.session_state.true_sentiment = None
        st.session_state.from_random = False

    review = st.text_area(
        "Wpisz recenzję filmu:",
        height=200,
        value=st.session_state.review,
        placeholder="Napisz swoją recenzję filmu tutaj",
        on_change=lambda: st.session_state.update({"from_random": False, "true_sentiment": None})
    )

    if st.session_state.from_random and st.session_state.true_sentiment is not None:
        true_sentiment_label = "✔ Pozytywny ✔" if st.session_state.true_sentiment == 1 else "❌ Negatywny ❌"
        st.write('Rzeczywisty sentyment recenzji: `%s`' % true_sentiment_label)

    col1, col2, col3 = st.columns(3)

    selected_option = None

    with col1:
        if st.button("🔍 Analizuj sentyment"):
            selected_option = "single"

    with col2:
        if st.button("🔍 Analizuj sentyment (tablica modeli)"):
            selected_option = "multiple"

    with col3:
        if st.button("🎲 Wylosuj recenzję"):
            if test_reviews:
                index = random.randint(0, len(test_reviews) - 1)
                st.session_state.review = test_reviews[index]
                st.session_state.true_sentiment = test_sentiments[index]
                st.session_state.from_random = True
                st.rerun()
            else:
                st.warning("Nie można załadować recenzji testowych.")

    if selected_option:
        if review.strip():
            if selected_option == "single":
                vectorizer, classifier = load_model_and_vectorizer(vectorizer_name, classifier_name)
                sentiment = analyze_sentiment(review, vectorizer, classifier, nlp)
                if sentiment == "Pozytywny":
                    st.success("Sentyment recenzji: ✔ Pozytywny ✔ ")
                else:
                    st.error("Sentyment recenzji: ❌ Negatywny ❌ ")
            elif selected_option == "multiple":
                results = analyze_sentiment_list_of_models(review, models, nlp)
                st.markdown("### Wyniki analizy sentymentu")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
        else:
            st.warning("Proszę wpisać recenzję do analizy.")


    st.markdown(
        "<br/> <p style='text-align: center;'>Ranking klasyfikatorów:</p>",
        unsafe_allow_html=True
    )

    st.dataframe(read_results())

if __name__ == "__main__":
    main()
