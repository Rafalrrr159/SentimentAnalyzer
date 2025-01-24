import os
import glob
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

def load_dataset(data_dir):
    data = []
    labels = []
    ids = []

    for file_path in glob.glob(os.path.join(data_dir, "pos", "*.txt")):
        with open(file_path, "r", encoding="utf-8") as file:
            data.append(file.read())
            labels.append(1)
            ids.append(os.path.basename(file_path).split("_")[0])

    for file_path in glob.glob(os.path.join(data_dir, "neg", "*.txt")):
        with open(file_path, "r", encoding="utf-8") as file:
            data.append(file.read())
            labels.append(0)
            ids.append(os.path.basename(file_path).split("_")[0])

    df = pd.DataFrame({"id": ids, "review": data, "sentiment": labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df

def preprocess_text(texts, nlp):
    processed_texts = []
    for doc in nlp.pipe(texts, batch_size=100):
        tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        processed_texts.append(" ".join(tokens))
    return processed_texts

def sentiment_analysis(dataset_dir):
    nlp = spacy.load("en_core_web_sm")

    if not os.path.exists("processed_train.csv"):
        train_data = load_dataset(os.path.join(dataset_dir, "train"))
        test_data = load_dataset(os.path.join(dataset_dir, "test"))
        train_data["review"] = preprocess_text(train_data["review"], nlp)
        test_data["review"] = preprocess_text(test_data["review"], nlp)
        train_data.to_csv("processed_train.csv", index=False)
        test_data.to_csv("processed_test.csv", index=False)
    else:
        train_data = pd.read_csv("processed_train.csv")
        test_data = pd.read_csv("processed_test.csv")

    # train_data["review"] = preprocess_text(train_data["review"], nlp)
    # test_data["review"] = preprocess_text(test_data["review"], nlp)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_data["review"])
    X_test = vectorizer.transform(test_data["review"])
    y_train = train_data["sentiment"]
    y_test = test_data["sentiment"]

    classifier = LogisticRegression(max_iter=100)
    classifier.fit(X_train, y_train)

    with open("tfidf_vectorizer.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)

    with open("logistic_model.pkl", "wb") as model_file:
        pickle.dump(classifier, model_file)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Dokładność modelu: {accuracy:.2f}")

    print("\nSzczegóły predykcji dla danych testowych:")
    results = pd.DataFrame({
        "id": test_data["id"],
        "review": test_data["review"],
        "true_sentiment": test_data["sentiment"],
        "predicted_sentiment": y_pred
    })
    for i, row in results.head(10).iterrows():
        true_sent = "Positive" if row["true_sentiment"] == 1 else "Negative"
        pred_sent = "Positive" if row["predicted_sentiment"] == 1 else "Negative"
        print(f"ID: {row['id']}")
        print(f"Review: {row['review'][:200]}...")
        print(f"True Sentiment: {true_sent}, Predicted Sentiment: {pred_sent}\n")

    return vectorizer, classifier, accuracy

if __name__ == "__main__":
    dataset_dir = input("Podaj ścieżkę do katalogu Large Movie Review Dataset: ")
    vectorizer, classifier, accuracy = sentiment_analysis(dataset_dir)
