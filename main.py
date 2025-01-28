import os
import glob
from pickle import FALSE

import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import MultinomialNB


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

    X_train, y_train = train_data["review"], train_data["sentiment"]
    X_test, y_test = test_data["review"], test_data["sentiment"]

    vectorizers = {
        "TF-IDF": (TfidfVectorizer(), [
            {
                "max_features": [500, 1000],
                "ngram_range": [(1, 1), (1, 2)]
            }
        ]),
        "CountVectorizer": (CountVectorizer(), [
            {
                "max_features": [500, 1000],
                "ngram_range": [(1, 1), (1, 2)]
            }
        ]),
        "HashingVectorizer": (HashingVectorizer(alternate_sign=False), [
            {
                "n_features": [500, 1000],
                "ngram_range": [(1, 1), (1, 2)]
            }
        ])
    }

    classifiers = {
        "LogisticRegression": (LogisticRegression(max_iter=2000), [
        {
            "penalty": ["l1", "l2"],
            "C": [0.8, 1.0, 1.2],
            "solver": ["liblinear", "saga"]
        }]),
        "NaiveBayes": (MultinomialNB(), [
        {
            "alpha": [1.2, 1.0, 0.8, 0.5, 0.2],
            "fit_prior": [True, False]
        }]),
        "LinearSVC": (LinearSVC(max_iter=2000), [
        {
            "penalty": ["l1", "l2"],
            "C": [0.8, 1.0, 1.2],
        }])
    }

    all_results = []
    for vec_name, (vectorizer, vec_params_grid) in vectorizers.items():
        print(f"Przetwarzanie wektorów za pomocą {vec_name}")
        for vec_params in ParameterGrid(vec_params_grid):
            print(f"Przetwarzanie wektorów za pomocą {vec_name} z parametrami: {vec_params}")

            vectorizer.set_params(**vec_params)

            X_train = vectorizer.fit_transform(train_data["review"])
            X_test = vectorizer.transform(test_data["review"])

            vectorizer_filename = f"{vec_name}_{vec_params}.pkl".replace(":", "").replace("{", "").replace(
                "}", "").replace("'", "").replace(",", "").replace(" ", "_")
            with open(vectorizer_filename, "wb") as vec_file:
                pickle.dump(vectorizer, vec_file)

            for clf_name, (clf, params_grid) in classifiers.items():
                print(f"Testowanie: {vec_name} + {vec_params} + {clf_name}")

                for params in ParameterGrid(params_grid):
                    model = clf.set_params(**params)
                    model.fit(X_train, y_train)

                    train_accuracy = accuracy_score(y_train, model.predict(X_train))
                    test_accuracy = accuracy_score(y_test, model.predict(X_test))

                    params_str = "_".join([f"{key}={value}" for key, value in params.items()])
                    model_name = f"{vec_name}_{vec_params}_{clf_name}_{params_str}.pkl".replace(":", "").replace("{", "").replace(
                        "}", "").replace("'", "").replace(",", "").replace(" ", "_")
                    with open(model_name, "wb") as file:
                        pickle.dump(model, file)

                    all_results.append({
                        "Vectorizer": vec_name,
                        "Vectorizer parameters": vec_params,
                        "Classifier": clf_name,
                        "Classifier parameters": params,
                        "Train Accuracy": train_accuracy,
                        "Test Accuracy": test_accuracy,
                        "Model File": model_name
                    })

    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv("all_results.csv", index=False)
    print("Wszystkie wyniki zapisano do pliku 'all_results.csv'.")
    return all_results_df

if __name__ == "__main__":
    dataset_dir = input("Podaj ścieżkę do katalogu Large Movie Review Dataset: ")
    #vectorizer, classifier, accuracy = sentiment_analysis(dataset_dir)
    results = sentiment_analysis(dataset_dir)
