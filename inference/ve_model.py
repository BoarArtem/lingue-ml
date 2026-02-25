from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format(
    "../inference/crawl-300d-2M.vec",
    binary=False
)

wv.save("crawl_fasttext.kv")