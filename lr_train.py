import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

from functions import get_sentence_embedding


def gen_train_data(ref_data_path, embed_tokenizer, embed_model, test_size=0.2):
    ref = pd.read_excel(ref_data_path)

    data = {i:[] for i in ["text", "label"]}
    for i in range(ref.shape[0]):
      if ref.iloc[i]['speaker'] == "Teacher":
        label = 0
      elif ref.iloc[i]['speaker'] == "Kid":
        label = 1
      else:
        continue
      data['text'].append(ref.iloc[i]['transcript'].strip())
      data['label'].append(label)

    X = [get_sentence_embedding(text, embed_tokenizer, embed_model) for text in data['text']]
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, y_train, X_test, y_test