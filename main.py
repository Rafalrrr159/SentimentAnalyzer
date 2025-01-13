import os
import glob
import pandas as pd

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

if __name__ == "__main__":
    dataset_dir = input("Podaj ścieżkę do katalogu Large Movie Review Dataset: ")
    train_data = load_dataset(os.path.join(dataset_dir, "train"))
    test_data = load_dataset(os.path.join(dataset_dir, "test"))
    print("Załadowano dane treningowe:")
    print(train_data.head())
    print("Załadowano dane testowe:")
    print(test_data.head())
