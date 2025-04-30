from model_helper_funcs import read_labeled_sentences, train_and_save_model

camper_data = read_labeled_sentences("camper_data.txt")
conviction_data = read_labeled_sentences("conviction_data.txt")
deed_data = read_labeled_sentences("deed_data.txt")

train_and_save_model(camper_data, "camper", "camper_model.pkl")
train_and_save_model(conviction_data, "conviction", "conviction_model.pkl")
train_and_save_model(deed_data, "deed", "deed_model.pkl")
