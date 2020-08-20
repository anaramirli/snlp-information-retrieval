from baseline_doc_retrieval import *
from rank_bm25 import BM25Okapi
import nltk


def rerank_bm25(queries, documents, tokenized_docs):
    """
    Rank documents using BM25 method.
    Return top 50 documents for all queries.

    :param queries(list): contains list of tokenized queries -> [['what', 'does', 'the', 'peugeot', 'company', 'manufacture'], ...]
    :param documents(list): raw documents as strings for each query -> [['A Malaysian English', '...'], ...]
    :param tokenized_docs(list): tokenized documents for each query -> [[['a', 'malaysian', 'english', '...'], ...]
    :return (list): contains lists with ranked raw documents as strings -> [['It is quite windy in London', '...'], ...]
    """

    top_50_raw = list()
    for q, docs_raw, docs_tokens in zip(queries, documents, tokenized_docs):
        bm25 = BM25Okapi(docs_tokens)
        top_ranked = bm25.get_top_n(q, docs_raw, n=50)
        top_50_raw.append(top_ranked)

    return top_50_raw


# def sort_docno(scores, document_numbers):
#
#     """
#     Sort the document numbers by descending scores
#
#     :param document_numbers(list): contains document number as strings
#     :param scores(list): contains float numbers
#     :return (list): sorted document numbers by scores in descending order
#     """
#
#     return [no for score, no in sorted(zip(scores, document_numbers))]

if __name__ == '__main__':

    # Preprocess queries
    articles = IRModel('data\\trec_documents.xml')
    queries = articles.extract_queries("data\\test_questions.txt") # list of queries as strings
    queries_tokenized = list()  # [['what', 'does', 'the', 'peugeot', 'company', 'manufacture'], ...]
    for q in queries:
        queries_tokenized.append(articles.preprocess_str(q))
    # Extract answers to all queries
    answers = articles.extract_answers("data\\patterns.txt")    # [["Young"], ["405", "automobiles?", "diesel\s+motors?" ],...]

    # 2a) Use BASELINE model and get the top 1000 documents for each query
    top_1000_raw = list()   # top 1000 documents for all queries -> [['JOHN LABATT, the Canadian food and beverage group,...', '...'],...]
    top_1000_tokenized = list()
    for q in queries:
        scores = articles.similarity_scores(q)[:1000]  # top 1000 documents for each query -> [(document number, score),..]
        docno = [no for no, score in scores]
        documents = articles.find_raw_document(docno)  # Get raw document content by document number -> ['JOHN LABATT, the Canadian food and beverage group,...', '...']
        top_1000_raw.append(documents)
        tokenized = list()
        for doc in documents:
            tokens = articles.preprocess_str(doc)
            tokenized.append(tokens)
        top_1000_tokenized.append(tokenized)

    print("The top 1000 documents ranked with the Baseline model are prepared....")

    # 2b) Use BM25 to get top 50 documents based on the top 1000 documents returned from baseline
    top_50_raw = rerank_bm25(queries_tokenized, top_1000_raw, top_1000_tokenized)
    # Test BM25 results with mean of the precisions
    print('Calculating Mean of Precisions for the top 50 documents ranked with BM25...')
    # precision mean:  0.104
    print('\nMean of Precisions for the top 50 documents ranked with BM25:', articles.precisions_mean(queries, answers, top_50_raw))

    # 3a) Split the top 50 documents into sentences.
    top_50_doc2sent_raw = list()
    top_50_doc2sent_tokenized = list()
    for docs in top_50_raw:   # loop for documents for a given query
        sents_raw = list()  # documents are strings
        sents_tokenized = list()    # documents are list of tokens
        for d in docs:
            splitted_sents = nltk.sent_tokenize(d)  # list of sents as strings
            for s in splitted_sents:    # preprocess, tokenize each sentence
                tokenized_s = articles.preprocess_str(s)
                sents_tokenized.append(tokenized_s)
                sents_raw.append(s)
        top_50_doc2sent_raw.append(sents_raw)
        top_50_doc2sent_tokenized.append(sents_tokenized)

    # 3b) Treat the sentences like documents to rank them and return the top 50 sentences ranked with BM25.
    top_50_raw_sents = rerank_bm25(queries_tokenized, top_50_doc2sent_raw, top_50_doc2sent_tokenized)
    for sents_raw, q, a in zip(top_50_raw_sents, queries, answers):
        print('Query: ', q)
        print('Answwers: ', a)
        print(sents_raw)
    # Mean of Precisions: 0.07
    print('\nMean of Precisions for the top 50 sentences ranked with BM25:', articles.precisions_mean(queries, answers, top_50_raw_sents))