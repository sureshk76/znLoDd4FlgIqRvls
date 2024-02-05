###  function for adding starred candidates
def add_starred_candidate_to_keywords(keywords, star_candidates_id_list, df):
    for i in star_candidates_id_list:
        keywords_list = (keywords.lower()).split()
        words_list = (df['job_title'][i].lower()).split()
        for word in words_list:
            if word not in keywords_list:
                keywords_list.append(word)
                keywords = ' '.join(keywords_list)
    return keywords