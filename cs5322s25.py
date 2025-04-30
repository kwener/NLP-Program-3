#first, use the training files to find the bert vector for the target word in each sentence
    #make sure training data is list of suples 0 is the sentence, 1 is the sense of the word
    #tag them accordingly, and then this is the dataset
#train model that predicts the sense (either 1 or 2)
"""
import torch
import joblib # to save the model 
import numpy as np
import random
from transformers import BertTokenizerFast, BertModel
from sklearn.model_selection import cross_val_score

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

    
    clf = LogisticRegression(max_iter=100)
    clf.fit(X_train, y_train)

    joblib.dump(clf, './camper_model.pkl') 

    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


def WSD_Test_camper(words):
    # Load the trained model
    clf = joblib.load('./camper_model.pkl')
    
    # Process each sentence and collect valid embeddings with their indices
    embeddings = []
    valid_indices = []
    
    for idx, sentence in enumerate(words):
        embedding = get_embedding(sentence, "camper")
        if embedding is not None:
            embeddings.append(embedding)
            valid_indices.append(idx)
    
    # Convert to numpy array if we have any valid embeddings
    if embeddings:
        X = np.stack(embeddings)
        predictions = clf.predict(X)
    else:
        # If no valid embeddings found, return random predictions as fallback
        predictions = np.random.choice([1, 2], size=len(words))
        return predictions.tolist()
    
    # Create full predictions list (1 or 2) for all sentences
    full_predictions = []
    pred_idx = 0
    for i in range(len(words)):
        if i in valid_indices:
            full_predictions.append(predictions[pred_idx])
            pred_idx += 1
        else:
            # For sentences where we couldn't get embedding, use random prediction as default
            full_predictions.append(random.choice([1, 2]))
    
    # nmpy makes it int.64 as a default
    return [int(pred) for pred in full_predictions] 

def WSD_Test_conviction(words):
    x = 0

def WSD_Test_deed(words):
    x = 0


camper_data = [
    # Sense 1: Van used for camping
    ("They rented a camper for their road trip through the mountains.", 2),
    ("The old camper had a tiny kitchenette and a fold-out bed.", 2),
    ("We parked the camper near the lake and set up a fire pit.", 2),
    ("Her dream is to convert a van into a cozy little camper.", 2),
    ("The camper was stocked with food and sleeping bags.", 2),
    ("They bought a new camper to travel across the country.", 2),
    ("The camper's roof popped up to create extra headroom.", 2),
    ("He lives in a camper and works remotely from national parks.", 2),
    ("They spent the night in a camper on the beach.", 2),
    ("The camper broke down on the side of the highway.", 2),
    ("They installed solar panels on the roof of their camper.", 2),
    ("I saw a vintage camper parked at the music festival.", 2),
    ("They towed a camper behind their truck during vacation.", 2),
    ("Their camper has a small bathroom and sleeping area.", 2),
    ("The family cooked breakfast inside their camper.", 2),
    ("The family will be so excited when they see their camper!", 2),

    # Sense 2: Child at a summer/day camp
    ("Every camper at the summer camp had to bring a sleeping bag.", 1),
    ("The counselor made sure each camper was accounted for.", 1),
    ("She became friends with another camper in her bunk.", 1),
    ("One camper got homesick and called their parents.", 1),
    ("The camper won a medal in the canoe race.", 1),
    ("Every camper received a t-shirt at the end of camp.", 1),
    ("The camper woke up early for morning activities.", 1),
    ("They assigned each camper to a different group.", 1),
    ("That camper performed a skit during talent night.", 1),
    ("The youngest camper in the group needed help tying shoes.", 1),
    ("Each camper was responsible for cleaning their area.", 1),
    ("The camper packed their things before heading home.", 1),
    ("As a returning camper, she helped welcome the new kids.", 1),
    ("One camper got stung by a bee on the nature hike.", 1),
    ("The campfire stories kept every camper entertained.", 1)
]

#camper_data_2 = read_labeled_sentences("/Users/kierstenwener/Downloads/prog3/camper.txt")
#camper_data_2= camper_data_2+camper_data
# Shuffle the list randomly
random.shuffle(camper_data)

# testing for the wsd: Camper, 1 is the can, 2 is the person
test_sentences = [
    "We slept in a camper van",
    "The camper built a fire",
    "I wish I had a camper",
    "The camper had two beds",
    "The camper ate hotdogs"
]

#do_camper_test(camper_data)

# testing for wsd: camper
predictions = WSD_Test_camper(test_sentences)
print(predictions)
"""""


from model_helper_funcs import get_embedding, load_and_predict

def WSD_Test_camper(sentences):
    return load_and_predict(sentences, "camper", "camper_model.pkl")

def WSD_Test_conviction(sentences):
    return load_and_predict(sentences, "conviction", "conviction_model.pkl")

def WSD_Test_deed(sentences):
    return load_and_predict(sentences, "deed", "deed_model.pkl")

test_sentences = [
  "After years of appeals, the conviction was finally overturned.",
"He stands by his conviction, even when it's unpopular.",
"The conviction resulted in a ten-year prison sentence.",
"Her conviction about the need for education reform is inspiring.",
"He was released after the wrongful conviction was dismissed.",
"She argued her point with unwavering conviction.",
"The jury reached a conviction in under an hour.",
"He pursued his goals with quiet conviction.",
"His conviction for fraud shocked the entire community.",
"Despite criticism, she expressed her conviction clearly.",
"The conviction led to a lengthy prison term.",
"Her conviction that art can change lives was contagious.",
"DNA evidence was used to overturn the original conviction.",
"They admired her conviction and sense of purpose.",
"The judge handed down the conviction without hesitation.",
"She delivered her speech with deep conviction.",
"The conviction was based on eyewitness testimony.",
"He always acts according to his personal conviction.",
"That conviction ended his political career.",
"She followed her conviction instead of chasing popularity."


]
print(WSD_Test_conviction(test_sentences))

