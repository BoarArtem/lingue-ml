import gensim.downloader as api

wv = api.load("word2vec-google-news-300")

wv.save("google_news.kv")
