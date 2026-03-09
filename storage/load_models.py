from gensim.models import KeyedVectors

model = None

def load_model():
    global model
    model = KeyedVectors.load("app/model/crawl_fasttext.kv")

    return model