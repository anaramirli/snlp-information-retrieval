from baseline_doc_retrieval import *
from rank_bm25 import BM25Okapi
import nltk


def rerank_bm25(queries, documents, tokenized_docs, docno):
    """
    Rank documents using BM25 method.
    Return top 50 documents for all queries.

    :param queries(list): contains list of tokenized queries -> [['what', 'does', 'the', 'peugeot', 'company', 'manufacture'], ...]
    :param documents(list): documents for all queries -> [['a malaysian english', '...'], ...]
    :param tokenized_docs(list): tokenized documents for all queries -> [[['a', 'malaysian', 'english', '...'], ...]
    :param docno(list): document numbers of the top n documents -> [[document number1, document number,...],...]
    :return (list, list, list): contains lists with ranked documents as strings -> [['it is quite windy in london', '...'], ...]
                                contains lists with ranked tokenized documents ->  [[['it', 'is', 'quite', 'windy', 'in', 'london'], '...'], ...]
                                contains lists of document numbers sorted by ranking scores
    """
    top_50 = list()
    tokenized_top_50 = list()
    docno_top_50 = list()
    for q, docs, token_docs, no in zip(queries, documents, tokenized_docs, docno):
        bm25 = BM25Okapi(token_docs)
        top_ranked = bm25.get_top_n(q, docs, n=50)
        top_50.append(top_ranked)
        tokens = list()
        for doc in top_ranked:
            tokens.append(doc.split())
        tokenized_top_50.append(tokens)
        # Sort document number by ranking scores
        docno_top_50.append(sort_docno(bm25.get_scores(q)[:50], no))

    return top_50, tokenized_top_50, docno_top_50


def sort_docno(scores, document_numbers):

    """
    Sort the document numbers by descending scores

    :param document_numbers(list): contains document number as strings
    :param scores(list): contains float numbers
    :return (list): sorted document numbers by scores in descending order
    """

    return [no for score, no in sorted(zip(scores, document_numbers))]

# Preprocess queries
articles = IRModel('data\\trec_documents.xml')
queries = articles.extract_queries("data\\test_questions.txt") # list of queries as strings
tokenized_queries = list()  # list of tokenized queries -> [['what', 'does', 'the', 'peugeot', 'company', 'manufacture'], ...]
for q in queries:
    tokenized_queries.append(articles.preprocess_str(q))
# Extract answers to all queries
answers = articles.extract_answers("data\\patterns.txt")    # list of lists containing answers to the queries -> [["Young"], ["405", "automobiles?", "diesel\s+motors?" ],...]


# 2a) Use baseline model and get the top 1000 documents for each query
baseline_top_1000 = list()   # top 1000 documents for all queries -> [['a malaysian english', '...'], ...]
tokenized_baseline_1000 = list()
docno_1000 = list() # document numbers for the top 1000 documents
for q in queries:
    scores = articles.similarity_scores(q)[:1000]  # top 1000 documents for each query -> [(document number, score),..]
    docno = [no for no, score in scores]
    docno_1000.append(docno)
    documents = articles.find_document(docno)  # Get document content by document number -> [['a', 'malaysian', 'english',], ...]
    tokenized_baseline_1000.append(documents)
    docs = list()
    for doc in documents:
        docs.append(' '.join(doc))
    baseline_top_1000.append(docs)

print(" Top 1000 from Baseline are prepared....")


# # 2b) Use BM25 to get top 50 documents based on the top 1000 documents returned from baseline
# bm25_top_50, tokenized_bm25_top_50, docno_top_50 = rerank_bm25(tokenized_queries, baseline_top_1000, tokenized_baseline_1000)
# # BM25: Mean of the precisions
# print('Calculating mean of the precisions (BM25)...')
# print('\nprecision mean (BM25):', articles.precisions_mean(queries, answers, tokenized_bm25_top_50)) # precision mean:  0.1059


# 3a) Split the top 50 documents into sentences
# Get raw document contents by document numbers,; 50 document numbers for a given query
# Preprocessed documents have no periods because of word tokenization and thus the information about end of the sentence within a document gets lost.
top_50_docs = list() # [["raw document 1", "raw document 2. this is a raw document.",...], ...]
for numbers in docno_1000:
    docs = articles.find_raw_document(numbers[:50])
    top_50_docs.append(docs)

# Split documents with multiple sentences such that each sentence will be treated as a document.
top_50_doc2sent = list()
tokenized_top_50_doc2sent = list()
for docs in top_50_docs:   # loop for documents for a given query
    sents = list()  # documents are strings
    tokenized_sents = list()    # documents are list of tokens
    for d in docs:
        splitted_sents = nltk.sent_tokenize(d)
        for s in splitted_sents:    # preprocess each sentence
            tokenized_s = articles.preprocess_str(s)
            tokenized_sents.append(tokenized_s)
            sents.append(' '.join(tokenized_s))
    top_50_doc2sent.append(sents)
    tokenized_top_50_doc2sent.append(tokenized_sents)


#3b) Treat the sentences like documents to rank them and return the top 50 sentences.
top_50_sents, tokenized_top_50, _ = rerank_bm25(tokenized_queries, top_50_doc2sent, tokenized_top_50_doc2sent, docno_1000[:50])
for sents, q, a in zip(top_50_sents, queries, answers):
    print('Query: ', q)
    print('Answwers: ', a)
    print(sents)

print(top_50_sents)
print('\nprecision mean (BM25):', articles.precisions_mean(queries, answers, tokenized_top_50)) # precision mean (BM25): 0.05000000000000002