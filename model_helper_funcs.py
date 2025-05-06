#first, use the training files to find the bert vector for the target word in each sentence
    #make sure training data is list of suples 0 is the sentence, 1 is the sense of the word
    #tag them accordingly, and then this is the dataset
#train model that predicts the sense (either 1 or 2)
import torch
import joblib # to save the model 
import numpy as np
import random
from transformers import BertTokenizerFast, BertModel
from sklearn.model_selection import cross_val_score


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


def read_labeled_sentences(filename):
    data = []
    with open(filename, 'r') as file:
        start_reading = False
        for line in file:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip initial lines until we encounter a line starting with '1' or '2' followed by a space
            if not start_reading:
                # Check if the line starts with '1' or '2' followed by a space (or tab)
                if line[:1].isdigit() and line[1] in [' ', '\t']:
                    start_reading = True
                else:
                    continue  # still skipping the definition lines

            if start_reading:
                label_str, sentence = line.split(maxsplit=1)  # split only once to get the label and sentence
                label = int(label_str)
                data.append((sentence, label))  # Store sentence with its label
    return data

def read_sentences(filename):
    sentences = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                sentences.append(line)
    return sentences


def get_embedding(sentence, word):
    tokenized = tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    offsets = tokenized["offset_mapping"][0]  # batch dim = 1

    sentence_lower = sentence.lower()
    target_words = [word, word+'s']

    found = False
    for word in target_words:
        start_char = sentence_lower.find(word)
        print(start_char)
        if start_char != -1:
            end_char = start_char + len(word)
            found = True
            break
    if not found:
        print(f"No target word found in: {sentence}")
        print(word)
        return None
    
    target_token_indices = []
    for idx, (start, end) in enumerate(offsets.tolist()):
        if start <= start_char < end or start < end_char <= end or (start_char <= start and end <= end_char):
            target_token_indices.append(idx)

    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
    
    if target_token_indices:
        target_embeddings = last_hidden_states[0, target_token_indices, :]
        word_embedding = target_embeddings.mean(dim=0)
        return word_embedding  # shape: [hidden_dim]
    else:
        print(f"No token match found for '{word}' in sentence: {sentence}")
        return None
    
def make_dataset(target_word, training_data): 
    embeddings, labels = [], []
    for sentence, sense in training_data:
        embedding = get_embedding(sentence, target_word)
        if embedding is not None:
            embeddings.append(embedding.numpy())
            labels.append(sense)
    X = np.stack(embeddings)
    y = np.array(labels)

    return X, y

def train_and_save_model(data, word, model_save_path):
     X, y = make_dataset(word, data)
     clf = LogisticRegression(max_iter=1000)
     clf.fit(X, y)
     joblib.dump(clf, model_save_path)


def load_and_predict(sentences, word, model_path):
    clf = joblib.load(model_path)
    predictions, valid_indices = [], []
    embeddings = []

    for idx, sentence in enumerate(sentences):
        embedding = get_embedding(sentence, word)
        if embedding is not None:
            embeddings.append(embedding)
            valid_indices.append(idx)

    if embeddings:
        X = np.stack(embeddings)
        predictions = clf.predict(X)
    else:
        # If no valid embeddings found, return random predictions as fallback
        predictions = np.random.choice([1, 2], size=len(sentences))
        return predictions.tolist()
    
    full_predictions = []
    pred_idx = 0
    for i in range(len(sentences)):
        if i in valid_indices:
            full_predictions.append(predictions[pred_idx])
            pred_idx += 1
        else:
            # For sentences where we couldn't get embedding, use random prediction as default
            full_predictions.append(random.choice([1, 2]))
    
    # nmpy makes it int.64 as a default
    return [int(pred) for pred in full_predictions]

def save_result_to_file(word, result_array):

    file_name = f"result_{word}_kierstenwener.txt"

    # Write to file
    with open(file_name, "w") as f:
        for value in result_array:
            f.write(f"{value}\n")




    
