#make sure the model_helper_funcs and trained model files are in the same folder to run
from model_helper_funcs import load_and_predict, save_result_to_file, read_sentences

def WSD_Test_camper(sentences):
    return load_and_predict(sentences, "camper", "camper_model.pkl")

def WSD_Test_conviction(sentences):
    return load_and_predict(sentences, "conviction", "conviction_model.pkl")

def WSD_Test_deed(sentences):
    return load_and_predict(sentences, "deed", "deed_model.pkl")

'''
deed_test_sentences = read_sentences("deed_test.txt")
conviction_test_sentences = read_sentences("conviction_test.txt")
camper_test_sentences = read_sentences("camper_test.txt")

save_result_to_file("conviction", WSD_Test_conviction(conviction_test_sentences))
save_result_to_file("camper", WSD_Test_camper(camper_test_sentences))
save_result_to_file("deed", WSD_Test_deed(deed_test_sentences))
'''
