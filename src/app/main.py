from sklearn.metrics.pairwise import cosine_similarity
from utilities.access_data import df_v1
from utilities.re_ranking import add_starred_candidate_to_keywords
from features.build_features import get_bert_embeddings
from features.build_features import compare_and_rank

import pandas as pd

def main():
    keywords = 'aspiring human resources'
    df = df_v1.copy()
    star_candidates = []
    df_bert = compare_and_rank(get_bert_embeddings, keywords, df)
    print('Bert ranking: ', df_bert[df_bert['bert_fit'] > 0.5])

if __name__ == '__main__':
    main()