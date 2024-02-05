import torchtext
import re
from sklearn.metrics.pairwise import cosine_similarity
from utilities import access_data

df = access_data.df_v1.copy()

def simple_cleaning(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def use_model(string):
    glove = torchtext.vocab.GloVe(name='6B', dim=100)
    tokens = string.split()
    indices = [glove.stoi[token] for token in tokens if token in glove.stoi]
    vectors = glove.vectors[indices]
    vectors_array = vectors.numpy()
    embedding = vectors_array.mean(axis=0)
    return embedding


def get_similarity_scores(text_embd, kw_embd):
    similarity_scores = [cosine_similarity(embd.reshape(1, -1), kw_embd.reshape(1, -1))[0, 0] for embd in text_embd]
    return similarity_scores

def main():
    df['job_title'] = df['job_title'].apply(simple_cleaning)
    keyword = 'aspiring human resources'
    embeddings = df['job_title'].apply(use_model)
    text_embedding_list = [embedding for embedding in embeddings]
    keyword_array = use_model(keyword)
    similarity_scores = get_similarity_scores(text_embedding_list, keyword_array)
    df['gloVe_fit'] = similarity_scores
    df_gloVe = df.sort_values('gloVe_fit', ascending=False)
    print('Glove rankings: ', df_gloVe[df_gloVe['gloVe_fit'] > 0.8])

if __name__ == '__main__':
    main()