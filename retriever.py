import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

kb = pd.read_csv("data/knowledge_base.csv")
kb["doc"] = kb["category"] + " " + kb["title"] + " " + kb["solution"]

vectorizer = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
kb_matrix = vectorizer.fit_transform(kb["doc"])


def search_kb(ticket_text, top_k=5): 
    query = vectorizer.transform([ticket_text])
    scores = cosine_similarity(query, kb_matrix)[0]
    top = scores.argsort()[::-1][:top_k]
    results = kb.iloc[top].copy()
    results["score"] = scores[top].round(2)
    return results

