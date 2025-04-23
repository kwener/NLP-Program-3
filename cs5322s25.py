#first, use the training files to find the bert vector for the target word in each sentence
    #make sure training data is list of suples 0 is the sentence, 1 is the sense of the word
    #tag them accordingly, and then this is the dataset
#train model that predicts the sense (either 1 or 2)
import torch
from transformers import BertTokenizerFast, BertModel
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


def get_camper_embedding(sentence):
    tokenized = tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    offsets = tokenized["offset_mapping"][0]  # batch dim = 1

    sentence_lower = sentence.lower()
    target_words = ["camper", "campers"]

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
    
    


def camper_model_training(training_data):
    x=0


def WSD_Test_camper(words):
    x = 0

def WSD_Test_conviction(words):
    x = 0

def WSD_Test_deed(words):
    x = 0

embedding = get_camper_embedding("The camper went away last summer")
print(embedding)