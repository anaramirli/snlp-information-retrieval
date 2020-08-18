from baseline_doc_retrieval import *
from rank_bm25 import BM25Okapi

# Preprocess documents
articles = IRModel('data\\trec_documents.xml')
tokenized_documents = articles.documents     # lists of tokenized documents -> [['a', 'malaysian', 'english',], ...]

# Preprocess queries
tokenized_queries = list()  # list of tokenized queries -> [['what', 'does', 'the', 'peugeot', 'company', 'manufacture'], ...]
queries = articles.extract_queries("data\\test_questions.txt") # list of queries as strings
for q in queries:
    tokenized_queries.append(articles.preprocess_str(q))

# Extract answers to all queries
answers = articles.extract_answers("data\\patterns.txt")    # list of lists containing answers to the queries -> [["Young"], ["405", "automobiles?", "diesel\s+motors?" ],...]

# Use baseline model and return the top 1000 documents given a query
baseline_top_1000 = list()   # top 1000 documents for all queries -> [['a malaysian english', '...'], ...]
for q in queries:
    scores = articles.similarity_scores(1000, q)  # top 1000 documents for each query -> [(document number, score),..]
    documents = articles.find_document(scores)  # Get document content by document number -> [['a', 'malaysian', 'english',], ...]
    docs = list()
    for doc in documents:
        docs.append(' '.join(doc))
    baseline_top_1000.append(docs)

print(" Top 1000 from Baseline are prepared....")

# Calculate scores via bm25
bm25 = BM25Okapi(baseline_top_1000)
bm25_top_50 = list()    # list of lists with ranked documents as strings -> [['It is quite windy in London', '...'], ...]
for q, docs in zip(tokenized_queries, baseline_top_1000):
   bm25 = BM25Okapi(docs)
   top_50 = bm25.get_top_n(q, docs, n=50)
   bm25_top_50.append(top_50)
   print(top_50)

# Return the top 50 documents based on the top 1000 documents returned in (a)
# for all queries
print(bm25_top_50)

