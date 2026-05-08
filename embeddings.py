from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

def get_embeddings(docs):
    return vectorizer.fit_transform(docs)

def get_query_embedding(query):
    return vectorizer.transform([query])