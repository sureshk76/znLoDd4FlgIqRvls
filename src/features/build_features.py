import pickle
import transformers
import torch
from sklearn.metrics.pairwise import cosine_similarity
from utilities.re_ranking import add_starred_candidate_to_keywords

model = transformers.BertModel.from_pretrained('bert-base-uncased')
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

def get_bert_embeddings(string, save_path=None, load_path=None):
    input_ids = tokenizer.encode_plus(string, add_special_tokens=True, return_tensors='pt')
    output = model(**input_ids)
    embedding = torch.mean(output.last_hidden_state, dim=1)
    if save_path is not None:
        with open(save_path, 'wb') as pkl:
            pickle.dump(embedding, pkl)
    return embedding

def compare_and_rank(embd, kw, df, st_cd = []):
    keywords = kw
    starred_candidates = st_cd
    if starred_candidates:
        keywords = add_starred_candidate_to_keywords(keywords, starred_candidates, df)
    txt_emb_bert = [embedding.detach().numpy() for embedding in df['job_title'].apply(embd)]
    kw_e_bert = embd(keywords)
    kw_emb_bert = kw_e_bert.detach().numpy()
    df['bert_fit'] = [cosine_similarity(txt_emb, kw_emb_bert).item() for txt_emb in txt_emb_bert]
    df_bert = df.sort_values('bert_fit', ascending=False)
    return df_bert
