from unittest.mock import MagicMock
from models.vocabulary_expander import VocabularyExpander

def test_vocabulary_expander_returns():
    fake_wv = MagicMock()
    fake_wv.most_similar.return_value = [("apple", 0.99), ("banana", 0.95)]

    ve = VocabularyExpander(fake_wv)
    result = ve.expand(["fruit"], topn=10)

    assert isinstance(result, list)
    assert all(isinstance(t, tuple) for t in result)
    assert all(isinstance(t[0], str) and isinstance(t[1], float) for t in result)

    fake_wv.most_similar.assert_called_once_with(positive=["fruit"], topn=10)
