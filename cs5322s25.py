#first, use the training files to find the bert vector for the target word in each sentence
    #make sure training data is list of suples 0 is the sentence, 1 is the sense of the word
    #tag them accordingly, and then this is the dataset
#train model that predicts the sense (either 1 or 2)
import torch
import numpy as np
import random
from transformers import BertTokenizerFast, BertModel
from sklearn.model_selection import cross_val_score

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




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
        if start_char != -1:
            end_char = start_char + len(word)
            found = True
            break
    if not found:
        print(f"No target word found in: {sentence}")
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
    
def make_camper_dataset(training): 
    camper_embeddings = []
    labels = []
    for sentence, sense in training:
        embedding = get_embedding(sentence, "camper")

        if embedding is not None:
            camper_embeddings.append(embedding.numpy())  # convert tensor to numpy
            labels.append(sense)
    X = np.stack(camper_embeddings)  # shape: (num_samples, 768)
    y = np.array(labels)             # shape: (num_samples,)
    return X, y

def do_camper_test(training):
    X, y = make_camper_dataset(training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scores = cross_val_score(LogisticRegression(max_iter=100), X, y, cv=5)
    print("Cross-validation scores:", scores)
    print("Mean accuracy:", scores.mean())

    '''
    clf = LogisticRegression(max_iter=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
'''

def WSD_Test_camper(words):
    x = 0

def WSD_Test_conviction(words):
    x = 0

def WSD_Test_deed(words):
    x = 0


camper_data = [
    # Sense 1: Van used for camping
    ("They rented a camper for their road trip through the mountains.", 1),
    ("The old camper had a tiny kitchenette and a fold-out bed.", 1),
    ("We parked the camper near the lake and set up a fire pit.", 1),
    ("Her dream is to convert a van into a cozy little camper.", 1),
    ("The camper was stocked with food and sleeping bags.", 1),
    ("They bought a new camper to travel across the country.", 1),
    ("The camper's roof popped up to create extra headroom.", 1),
    ("He lives in a camper and works remotely from national parks.", 1),
    ("They spent the night in a camper on the beach.", 1),
    ("The camper broke down on the side of the highway.", 1),
    ("They installed solar panels on the roof of their camper.", 1),
    ("I saw a vintage camper parked at the music festival.", 1),
    ("They towed a camper behind their truck during vacation.", 1),
    ("Their camper has a small bathroom and sleeping area.", 1),
    ("The family cooked breakfast inside their camper.", 1),
    ("The family will be so excited when they see their camper!", 1),

    # Sense 2: Child at a summer/day camp
    ("I would rather sleep with the campers", 2),
    ("The family likes the camper", 2),
    ("Every camper at the summer camp had to bring a sleeping bag.", 2),
    ("The counselor made sure each camper was accounted for.", 2),
    ("She became friends with another camper in her bunk.", 2),
    ("One camper got homesick and called their parents.", 2),
    ("The camper won a medal in the canoe race.", 2),
    ("Every camper received a t-shirt at the end of camp.", 2),
    ("The camper woke up early for morning activities.", 2),
    ("They assigned each camper to a different group.", 2),
    ("That camper performed a skit during talent night.", 2),
    ("The youngest camper in the group needed help tying shoes.", 2),
    ("Each camper was responsible for cleaning their area.", 2),
    ("The camper packed their things before heading home.", 2),
    ("As a returning camper, she helped welcome the new kids.", 2),
    ("One camper got stung by a bee on the nature hike.", 2),
    ("The campfire stories kept every camper entertained.", 2)
]

# Shuffle the list randomly
random.shuffle(camper_data)

do_camper_test(camper_data)