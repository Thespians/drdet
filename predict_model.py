from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from train_model import train 
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np 
import pandas as pd

def predict(string:str):
    text = string
    voc_size = 5000
    sent_length = 20
    corpus = preprocess(text)

    one_hot_rep = [one_hot(words,voc_size)for words in corpus]

    embedded = pad_sequences(one_hot_rep, padding = 'pre', maxlen = sent_length)

    txt_final = np.array(embedded)
    model = load_model("model.hdf5")
    pred = model.predict(txt_final) 
    if pred > 0.5:
        label = 1
    else:
        label = 0

    return label

def preprocess(string:str):
    ps = PorterStemmer()
    corpus = []
    rev = re.sub('[^a-zA-Z]', ' ', string)
    rev = rev.lower()
    rev = rev.split()
    rev = [ps.stem(word) for word in rev if not word in stopwords.words('english')]
    rev = ' '.join(rev)
    corpus.append(rev)

    return corpus

def update_dataset(text, label):
    
    df = pd.read_csv(r"D:\ozmenta\final\final\data\train.csv", index_col = 0)

    df.loc[len(df.index)] = [text,label]
    df.to_csv(r"D:\ozmenta\final\final\data\train.csv")

if __name__=='__main__':
    text = input("Enter Statement:")
    label = predict(text)
    print(label)
    update_dataset(text, label)
    train()