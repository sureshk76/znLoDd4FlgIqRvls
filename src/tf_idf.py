from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from utilities import access_data
### data ###
df_v1 = access_data.df_v1.copy()

documents = df_v1['job_title'].tolist()
keywords = 'Aspiring Human Resources'

def train_model():
    vectorizer = TfidfVectorizer()
    trained_text_vectors = vectorizer.fit_transform(documents)
    keyword_vector = vectorizer.transform([keywords])
    return trained_text_vectors, keyword_vector

def how_similar(txt_vectors, kw_vector):
    similarities = [cosine_similarity(txt_vector, kw_vector) for txt_vector in txt_vectors]
    similarities_list = []
    for i in similarities:
        similarities_list.append(i.item())
    return similarities_list


def main():
    text_vecs, kw_vec = train_model() 
    similarity_score = how_similar(text_vecs, kw_vec)
    df_v1['tfidf_fit'] = similarity_score
    df_tfidf = df_v1.sort_values('tfidf_fit', ascending=False)
    print('tf_idf ranked list: ', df_tfidf[df_tfidf['tfidf_fit'] > 0.5])

if __name__ == '__main__':
    main()