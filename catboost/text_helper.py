import re
import nltk
from string import punctuation
from pymorphy3 import MorphAnalyzer


def clean_text(text):
    text = re.sub(r"^[А-ЯЁ].{2,}\W{2}РИА Новости, [\w\s,]+\.", "", text)
    text = re.sub(r"^[А-ЯЁ].{2,}\W{2}РИА Новости\.", "", text)
    text = re.sub(r'http[^А-Я]+', ' ', text)
    text = re.sub(r'\d+', '1', text)
    text = re.sub(r'\xa0', ' ', text)
    text = text.translate(str.maketrans('', '', punctuation.replace('-', '') + '”„“«»†*\—/\\‘’'))
    text = re.sub(r'ё', 'е', text)
    text = re.sub(r'-(?=\s)', '', text) #we keep hyphens inside words, standalone ones are deleted         
    return text.lower().strip()

def tokenize(text, regex=re.compile("[а-яa-zё\d-]+")):
    try:
        tokens = regex.findall(text)
        # Tokens like "3D" or "G7" are not affected, while pure numeric tokens are removed
        tokens =  [(re.sub(r'^\d+$', '', token)) for token in tokens]
        tokens = [token for token in tokens if token.strip()]
        return tokens
    except:
        return []

def pymorphy_lemmatizer(tokens, pymorphy=MorphAnalyzer()):
    return [pymorphy.parse(token)[0].normal_form for token in tokens]

def remove_stopwords(lemmas, stopwords):
    return [w for w in lemmas if not w in stopwords]

def lemmatize(text, delete_stopwords=True, stopwords=None):
    tokens = tokenize(text)
    lemmas = pymorphy_lemmatizer(tokens)
    lemmas = [re.sub('ё', 'е', lemma) for lemma in lemmas]

    if delete_stopwords and stopwords is not None:
        lemmas = remove_stopwords(lemmas, stopwords=stopwords)

    return ' '.join(lemmas)


def preprocess_text(df, stopwords=None, delete_stopwords=True, make_lemmatization=True):
    df['content_clean'] = df.content.apply(clean_text)
    df['title_clean'] = df.title.apply(clean_text)
    #Combine the text features: the title and the content of every news article.
    df['text_clean'] = df['title_clean'] + ' ' + df['content_clean']
    if make_lemmatization:
        df['lemmas'] = df['text_clean'].swifter.apply(
            lambda x: lemmatize(x, stopwords=stopwords, delete_stopwords=delete_stopwords)
        )
        #df['title_lemmas'] = df.title_clean.swifter.apply(
        #    lambda x: lemmatize(x, stopwords=stopwords, delete_stopwords=delete_stopwords)
        #)
        #df['content_lemmas'] = = df.title_clean.swifter.apply(
        #    lambda x: lemmatize(x, stopwords=stopwords, delete_stopwords=delete_stopwords)
        #)
        #df['text_lemmas'] = df['title_clean'] + ' ' + df['content_clean']

    return df