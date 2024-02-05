import pandas as pd
import math
import re
from collections import Counter
from utilities import access_data

df_v1 = access_data.df_v1.copy()

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return float(numerator/denominator)
    
def text_to_vector(text):
    return Counter(WORD.findall(text))


keyword = 'Aspiring Human Resources'

def main():
    job_title_vectors = [text_to_vector(text) for text in df_v1['job_title']]
    keyword_vectors = text_to_vector(keyword)
    df_v1['bow_fit'] = [get_cosine(keyword_vectors, title_vector) for title_vector in job_title_vectors]
    df_bow = df_v1.sort_values('bow_fit', ascending=False)
    print('Bag of words ranked list: ', df_bow[df_bow['bow_fit'] > 0.5])

if __name__ == '__main__':
    main()