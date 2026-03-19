from gensim.models import KeyedVectors

model = None

def load_model():
    global model
    model = KeyedVectors.load("app/model/russian_word2vec.model")

    return model