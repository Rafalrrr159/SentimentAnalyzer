import os
import glob
import pandas as pd
import spacy

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

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    dataset_dir = input("Podaj ścieżkę do katalogu Large Movie Review Dataset: ")
    train_data = load_dataset(os.path.join(dataset_dir, "train"))
    test_data = load_dataset(os.path.join(dataset_dir, "test"))

    train_data["review"] = preprocess_text(train_data["review"], nlp)
    test_data["review"] = preprocess_text(test_data["review"], nlp)

