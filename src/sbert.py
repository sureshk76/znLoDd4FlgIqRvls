import transformers
import torch
from sklearn.metrics.pairwise import cosine_similarity

from utilities import access_data
from utilities.re_ranking import add_starred_candidate_to_keywords

model_sbert = transformers.AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
tokenizer_sbert = transformers.AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def string_to_sbert_embedding(string):
    input_ids = tokenizer_sbert.encode_plus(string, add_special_tokens=True, return_tensors='pt')
    output = model_sbert(**input_ids)
    embedding = torch.mean(output.last_hidden_state, dim=1)
    return embedding

def main():
    df = access_data.df_v1.copy()
    keywords = 'aspiring human resources'
    starred_candidates = [78]
    if starred_candidates:
        keywords = add_starred_candidate_to_keywords(keywords, starred_candidates, df)
    txt_emb_sbert = [embedding.detach().numpy() for embedding in df['job_title'].apply(string_to_sbert_embedding)]
    kw_e_sbert = string_to_sbert_embedding(keywords)
    kw_emb_sbert = kw_e_sbert.detach().numpy()   
    df['sbert_fit'] = [cosine_similarity(txt_emb, kw_emb_sbert).item() for txt_emb in txt_emb_sbert]
    df_sbert = df.sort_values('sbert_fit', ascending=False)
    print('Sbert ranking:', df_sbert[df_sbert['sbert_fit'] > 0.5])

if __name__ == '__main__':
    main()