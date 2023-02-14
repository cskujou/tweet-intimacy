from tqdm import tqdm
import pandas as pd

uncleaned_data = pd.read_csv("train.csv")

with tqdm(total=len(uncleaned_data)) as pbar:
    for i in range(len(uncleaned_data)):
        words = []
        for word in uncleaned_data["text"][i].split(" "):
            if word.startswith("@") and len(word) > 1:
                word = "@user"
            words.append(word)
        uncleaned_data["text"][i] = " ".join(words)
        pbar.update(1)

uncleaned_data.to_csv("train_cleaned.csv")