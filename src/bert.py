import transformers
import torch
from sklearn.metrics.pairwise import cosine_similarity

from utilities import access_data
from utilities.re_ranking import add_starred_candidate_to_keywords

model = transformers.BertModel.from_pretrained('bert-base-uncased')
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

def string_to_bert_embedding(string):
    input_ids = tokenizer.encode_plus(string, add_special_tokens=True, return_tensors='pt')
    output = model(**input_ids)
    embedding = torch.mean(output.last_hidden_state, dim=1)
    return embedding

def main():
    keywords = 'aspiring human resources'
    df = access_data.df_v1.copy()
    starred_candidates = []
    if starred_candidates:
        keywords = add_starred_candidate_to_keywords(keywords, starred_candidates, df)
    txt_emb_bert = [embedding.detach().numpy() for embedding in df['job_title'].apply(string_to_bert_embedding)]
    kw_e_bert = string_to_bert_embedding(keywords)
    kw_emb_bert = kw_e_bert.detach().numpy()
    df['bert_fit'] = [cosine_similarity(txt_emb, kw_emb_bert).item() for txt_emb in txt_emb_bert]
    df_bert = df.sort_values('bert_fit', ascending=False)
    print('Bert ranking: ', df_bert[df_bert['bert_fit'] > 0.5])

if __name__ == '__main__':
    main()