import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

dataset_path = 'D:/Personal/University/کارشناسی/Project/Implementation/data/text/emotions.csv'

def load_data(data_path):
    data = pd.read_csv(data_path)
    return train_test_split(data, train_size=0.8)

def tokenize_data(train_data, test_data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(list(train_data['text']), padding=True, truncation=True, max_length=128)
    test_encodings = tokenizer(list(test_data['text']), padding=True, truncation=True, max_length=128)
    return train_encodings, test_encodings

data = pd.read_csv(dataset_path)

train_data, test_data = load_data(dataset_path)

# print(train_data.head)

# print(data.columns)

encoded_train, encoded_test = tokenize_data(train_data, test_data)

print(encoded_test[1 : 3])
