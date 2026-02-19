from gensim.models import KeyedVectors

wv = KeyedVectors.load("../inference/google_news.kv", mmap="r")

def pretrained_model(arr):
    return wv.most_similar(
        positive=arr
    )
