from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utilities import access_data

df = access_data.df_v1.copy()

def tokenized_corpus(text):
    tokens = text.split()
    return tokens

def get_model(tokenized_list):
    model = Word2Vec(tokenized_list, vector_size=100, window=5, min_count=2, workers=4)
    model.save('word2vec.model', )
    model = Word2Vec.load("word2vec.model")
    return model

def get_embedding(model, string):
    tokens = string.split()
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    embedding = np.mean(vectors, axis=0)
    return embedding

def main():
    keywords = 'aspiring human resources'
    toks = df['job_title'].apply(tokenized_corpus)
    tokenized_list = toks.tolist()
    model = get_model(tokenized_list)
    text_embeddings = df['job_title'].apply(lambda x: get_embedding(model, x))
    kw_embeddings = get_embedding(model, keywords)
    similarity_scores = [cosine_similarity(array.reshape(1, -1), kw_embeddings.reshape(1, -1))[0, 0] for array in text_embeddings]
    df['w2v_fit'] = similarity_scores
    df_w2v = df.sort_values('w2v_fit', ascending=False)
    print('Word 2 Vec ranking: ', df_w2v)

if __name__ == '__main__':
    main()