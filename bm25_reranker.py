from baseline_doc_retrieval import *
from rank_bm25 import BM25Okapi


# Preprocess queries
articles = IRModel('data\\trec_documents.xml')
queries = articles.extract_queries("data\\test_questions.txt") # list of queries as strings
tokenized_queries = list()  # list of tokenized queries -> [['what', 'does', 'the', 'peugeot', 'company', 'manufacture'], ...]
for q in queries:
    tokenized_queries.append(articles.preprocess_str(q))

# Extract answers to all queries
answers = articles.extract_answers("data\\patterns.txt")    # list of lists containing answers to the queries -> [["Young"], ["405", "automobiles?", "diesel\s+motors?" ],...]

# Use baseline model and get the top 1000 documents for each query
baseline_top_1000 = list()   # top 1000 documents for all queries -> [['a malaysian english', '...'], ...]
tokenized_baseline_1000 = list()
for q in queries:
    scores = articles.similarity_scores(q)[:1000]  # top 1000 documents for each query -> [(document number, score),..]
    documents = articles.find_document(scores)  # Get document content by document number -> [['a', 'malaysian', 'english',], ...]
    tokenized_baseline_1000.append(documents)
    docs = list()
    for doc in documents:
        docs.append(' '.join(doc))
    baseline_top_1000.append(docs)
print(" Top 1000 from Baseline are prepared....")

# Use BM25 to get top 50 documents based on the top 1000 documents returned from baseline
bm25_top_50 = list()    # list of lists with ranked documents as strings -> [['It is quite windy in London', '...'], ...]
tokenized_bm25_top_50 = list()
for q, docs, tokenized_docs in zip(tokenized_queries, baseline_top_1000, tokenized_baseline_1000):
    bm25 = BM25Okapi(tokenized_docs)
    top_50 = bm25.get_top_n(q, docs, n=50)
    bm25_top_50.append(top_50)
    tokens = list()
    for doc in top_50:
        tokens.append(doc.split())
    tokenized_bm25_top_50.append(tokens)

# BM25: Mean of the precisions
print('Calculating mean of the precisions (BM25)...')
print('\nprecision mean (BM25):', articles.precisions_mean(queries, answers, tokenized_bm25_top_50)) # precision mean:  0.1059

