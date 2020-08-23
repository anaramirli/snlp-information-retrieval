from baseline_doc_retriever import *
from rank_bm25 import BM25Okapi
import nltk
import optparse

class BM25Model(object):
    """
    This class is used to rank documents using BM25 method. in order to return top 50 documents for all queries.
    """

    def rerank_bm25(self, queries, documents, tokenized_docs, k1=0.75, b=0.05):
        """
        Rank documents using BM25 method.
        Return top 50 documents for all queries.

        :param queries(list): contains list of tokenized queries -> [['what', 'does', 'the', 'peugeot', 'company', 'manufacture'], ...]
        :param documents(list): raw documents as strings for each query -> [['A Malaysian English', '...'], ...]
        :param tokenized_docs(list): tokenized documents for each query -> [[['a', 'malaysian', 'english', '...'], ...]
        :param k1(float): controls how quickly an increase in term frequency results in term-frequency saturation
        :param b(float): controls how much effect field-length normalization should have
        :return (list): contains lists with ranked raw documents as strings -> [['It is quite windy in London', '...'], ...]
        """

        top_50_raw = list()
        for q, docs_raw, docs_tokens in zip(queries, documents, tokenized_docs):
            bm25 = BM25Okapi(corpus=docs_tokens, k1=k1, b=b)
            top_ranked = bm25.get_top_n(q, docs_raw, n=50)
            top_50_raw.append(top_ranked)

        return top_50_raw


if __name__ == '__main__':
    # Parse command line arguments
    optparser = optparse.OptionParser()
    optparser.add_option("-d", dest="data", default="data\\trec_documents.xml", help="Path to raw documents.")
    optparser.add_option("-q", dest="queries", default="data\\test_questions.txt", help="Path to raw queries.")
    optparser.add_option("-a", dest="answers", default="data\\patterns.txt", help="Path to answer patterns.")
    (opts, _) = optparser.parse_args()
    path2docs = opts.data
    path2queries = opts.queries
    path2answers = opts.answers

    # Initialize Information Retrieval Model
    articles = IRModel(path2docs)
    queries = articles.extract_queries(path2queries) # list of queries as strings
    queries_tokenized = list()  # [['what', 'does', 'the', 'peugeot', 'company', 'manufacture'], ...]
    for q in queries:
        queries_tokenized.append(articles.preprocess_str(q))
    # Extract answers to all queries
    answers = articles.extract_answers(path2answers)    # [["Young"], ["405", "automobiles?", "diesel\s+motors?" ],...]
    
    # Initialize BM25Model
    bm25model = BM25Model()

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
    top_50_raw = bm25model.rerank_bm25(queries_tokenized, top_1000_raw, top_1000_tokenized, k1=0.75, b=0.05)

    # Evaluate BM25 model with mean of precisions and MRR
    print('\nEvaluating the performance of the BM25 model...')
    # Mean of Precisions:  0.122
    print('\nMean of Precisions for the top 50 documents ranked with BM25:', articles.precisions_mean(queries, answers, top_50_raw))
    # Mean reciprocal rank: 0.712
    print("\nMean reciprocal rank for the top 50 documents ranked with BM25: ", articles.mean_reciprocal_rank(answers, top_50_raw))

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
    top_50_raw_sents = bm25model.rerank_bm25(queries_tokenized, top_50_doc2sent_raw, top_50_doc2sent_tokenized, k1=0.05, b=0.05)

    # Check contents of raw queries, answers and sentences
    # for sents_raw, q, a in zip(top_50_raw_sents, queries, answers):
    #     print('\nQuery: ', q)
    #     print('Answwers: ', a)
    #     print('Raw sentences/documents: ', sents_raw, '\n')

    # Mean of Precisions: 0.081
    print('\nMean of Precisions for the top 50 sentences ranked with BM25:', articles.precisions_mean(queries, answers, top_50_raw_sents))

    # 3c) Evaluate the performance of the model using the mean reciprocal rank function (MRR) on the test queries Q
    # Mean reciprocal rank: 0.525
    print('\nMean reciprocal rank for the top 50 sentences ranked with BM25:', articles.mean_reciprocal_rank(answers, top_50_raw_sents))