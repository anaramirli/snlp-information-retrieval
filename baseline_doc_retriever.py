from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
import math
from collections import Counter
import operator
import numpy as np
import re
import optparse


class IRModel:
    """
    This class calculates tf-idf scores in order to return n most relevant documents given a query.
    """

    def __init__(self, path2docs):
        """
        :param docs(str): path to the documents
        """
        # Prepare documents
        self.docno, self.raw_documents = self.extract_text(path2docs)
        self.documents = self.preprocess(self.raw_documents)
        self.vocab = self.get_vocab(self.documents)
        self.N = len(self.documents)        # total number of documents

    def extract_text(self, path2docs):
        """
        Extract document number and text from .xml files

        :param path2docs(str):
        :return (list, list): document numbers, documents as strings
                               -> [' LA123189-0111 ', ' LA123189-0133 ',...] , [ "Sudden heart rejection...", "...",...]
        """
        documents = open(path2docs, encoding='utf-8').read()

        soup = BeautifulSoup(documents, 'lxml')
        doc_numbers = list()
        text = list()
        for docno, content in zip(soup.find_all('docno'), soup.find_all('text')):
            doc_numbers.append(docno.text)
            stripped_content = content.text.replace('\n', '') 
            text.append(stripped_content)
        return doc_numbers, text

    def preprocess(self, text):
        """
        tokenize, lower-case, remove punctuation

        :param text(list): contains documents as strings -> [ "Sudden heart rejection...",...]
        :return (list): contains lists of tokens -> [['a', 'malaysian', 'english',], ...]
        """
        tokenizer = RegexpTokenizer(r'\w+')
        preprocessed = list()
        for t in text:
            t = t.lower()
            preprocessed.append(tokenizer.tokenize(t))
        return preprocessed

    def preprocess_str(self, sentence):
        """
        Pre-process one sentence string; process for query

        :param sentence(str):
        :return(list): list of tokens -> ['what', 'does', 'the', 'peugeot', 'company', 'manufacture']
        """
        tokenizer = RegexpTokenizer(r'\w+')
        sentence = tokenizer.tokenize(sentence.lower())
        return sentence

    def get_vocab(self, text):
        """
        Create vocabulary of the documents

        :param text(list): contains lists of tokens -> [['a', 'malaysian', 'english',], ...]
        :return(set):  vocabulary
        """
        vocab = list()
        for doc in text:
            vocab.extend(doc)
        set(vocab)
        return set(vocab)

    def idf(self, term):
        """
        idf score for one term
        idf(term_i) = log(total number of docs/ number of docs that contain the term_i)

        :param term(str):
        :return(float)
        """
        n_term = 0
        for doc in self.documents:
            if term in doc:
                n_term += 1
        if n_term == 0:
            return 0
        else:
            return math.log(self.N / n_term)

    def tf(self, term, doc):
        """
        tf scores for one document.
        tf(term_i, doc) = number of times term_i appears in the document
                        / number of times the most frequent term of the document appears in the document

        :param term(str):
        :param doc(list): list of tokens
        :return(float):
        """
        terms_in_doc = Counter(doc) # {term_i: int }
        max_term = max(terms_in_doc.values()) # number of times the most frequent term of the document appears in the document

        return terms_in_doc[term] / max_term

    def get_vector(self, terms, document, idf_scores):
        """
        Creat vector for each document/query.
        Length of vector is the same as the length of query.

        :param terms(list): list of tokens from query
        :param document(list):list of tokens from document/query
        :param idf_scores(list): list of idf values for the terms of the query
        :return(list): contains float numbers
        """

        vector = list()
        for term, idf in zip(terms, idf_scores):
            tf_idf = self.tf(term, document) * idf
            vector.append(tf_idf)

        return vector

    def similarity_scores(self, query):
        """
        Return all relevant documents for the given query

        :param query(str):
        :return(list): list of tuples; contains document number with its score in descending order
                        -> [(document number, score),..]
        """
        # pre-process query like a document
        query = self.preprocess_str(query)  # list of tokens
        # Store idf_scores for the terms occurring in the query
        idf_scores = [self.idf(term) for term in query]
        # Get vector for query
        query_vec = self.get_vector(query, query, idf_scores)

        # Get similarity scores for each document
        similarity_socres = dict()
        for doc, no in zip(self.documents, self.docno):  # look in pre-processed documents; initialized in tf_idf_weights method
            doc_vec = self.get_vector(query, doc, idf_scores)
            
            # caculate the cosine similarity
            if np.dot(query_vec, doc_vec)!=0:
                cosine_sim = np.dot(query_vec, doc_vec) / \
                (np.sqrt(np.sum(np.square(query_vec))) * np.sqrt(np.sum(np.square(doc_vec))))
            else: cosine_sim = 0
            
            similarity_socres[no] = cosine_sim

        # Sort in descending order
        similarity_socres = sorted(similarity_socres.items(), key=operator.itemgetter(1), reverse=True)

        return similarity_socres

    def extract_queries(self, path2queries):
        """
        Extract queries from the descriptions <desc>

        :param path2queries:
        :return(list): contains queries as strings
        """

        queries = open(path2queries, encoding='utf-8').read()
        soup = BeautifulSoup(queries, 'lxml')
        queries = list()
        for q in soup.find_all('desc'):
            q = q.text.split()      # Get rid of 'Description:'
            del q[0]
            queries.append(' '.join(q))
        return queries

    def extract_answers(self, path2answers):
        """
        Answers to the corresponding queries in patterns.txt
        Answers are expressed as regex patterns

        :param path2answers(str):
        :return(list): list of lists containing answers to the queries
                        -> [["Young"], ["405", "automobiles?", "diesel\s+motors?" ],...]
        """
        no = 1
        answers = list()
        patterns = list()
        with open(path2answers, encoding='utf-8') as f:
            for line in f:
                line = line.split()
                if int(line[0]) > no:
                    answers.append(patterns)
                    patterns = list()
                    no += 1
                patterns.extend(line[1:])
        answers.append(patterns)
        
        return answers

    def is_relevant(self, answers, retrieved_documents):
        """
        Count relevant documents for the precision calculation

        :param answers(list): contains regex patterns as strings
                                -> [["Young"], ["405", "automobiles?", "diesel\s+motors?" ],...]
        :param retrievend_documents(list): raw documents as strings
                                            -> ['JOHN LABATT, the Canadian food and beverage group,...', '...']
        :return(int): number of relevant documents
        """
        relevant = 0
        # Check whether one of the answers is in the document
        for doc in retrieved_documents:
            if any(re.search(pattern, doc) for pattern in answers):
                relevant += 1

        return relevant

    def precision(self, answers, documents, r=50):
        """
        Calculate precision for each query

        :param answers(list): contains strings of regex patterns
        :param documents(list): contains strings of raw documents
                                -> ['JOHN LABATT, the Canadian food and beverage group,...', '...']
        :param r(int): percentage of relevant documents from the top n retrieved documents
        :return(float): precision value for one query
        """    
        n_relevant = self.is_relevant(answers, documents[:r])     # number of relevant and retrieved documents
        precision = n_relevant / r
        
        return precision

    def precisions_mean(self, queries, answers, retrieved_docs, r=50):
        """
        precision = # relevant and retrieved documents / # retrieved documents
        A document is relevant if it contains the answer
        Accept only tokenized documents

        :param queries(list): contains strings of queries
        :param answers(list): list of lists with regex patterns as strings
        :param retrieved_docs(list): ranked retrieved documents for all queries
                                        -> [['JOHN LABATT, the Canadian food and beverage group,...', '...'],...]
        :param r(int): number of top most relevant documents
        :return(float): mean precision
        """
        precisions = list()
        for q, a, docs in zip(queries, answers, retrieved_docs):
            precision = self.precision(a, docs, r)
            precisions.append(precision)

        precisions_mean = sum(precisions) / len(precisions)

        return precisions_mean
        
    def mean_reciprocal_rank(self, answers, retrieved_docs):
        """
        Evaluate the performance of your model using the mean reciprocal rank function (MRR)
        
        :param answers(list): lists with list of regex patterns as string for each query
        :param retrieved_docs(list): ranked retrieved documents for all queries
                                    -> [['JOHN LABATT, the Canadian food and beverage group,...', '...'],...]
        :return(float): mrr score
        """
        
        mrr = 0
        # for each query we have one answer, hence, number wise iterating through the answers is as same as iterating through the query 
        for query_i, ans in enumerate(answers):
            for doc_rank, doc in enumerate(retrieved_docs[query_i]):
                # get the rank of the first relevant sentence for the query q and then do the caculation
                if any(re.search(pattern, doc) for pattern in ans):
                    mrr += 1/(doc_rank+1)
                    break
                    # if none of the proposed retrived docs are correct, reciprocal rank is 0, thus we do nothing
        mrr /= len(answers)

        return mrr

    def find_raw_document(self, document_numbers):
        """
        Find raw, unprocessed document contents by the documents number
        :param document_numbers(list): [document number1, ...]
        :return(list): raw documents, one document represented as one string
                        -> ['JOHN LABATT, the Canadian food and beverage group,...', '...']
        """
        documents = list()
        for doc_no in document_numbers:
            idx = self.docno.index(doc_no)
            doc_content = self.raw_documents[idx]
            documents.append(doc_content)

        return documents


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
    queries = articles.extract_queries(path2queries)
    answers = articles.extract_answers(path2answers)

    # Rank total documents with the baseline model
    retrieved_docs_50 = list()
    retrieved_docs_total = list()
    for q in queries:
        sim_scores = articles.similarity_scores(q)
        docno = [no for no, score in sim_scores]
        docs_50 = articles.find_raw_document(docno[:50])
        docs_total = articles.find_raw_document(docno)
        retrieved_docs_50.append(docs_50)
        retrieved_docs_total.append(docs_total)
    print('The documents are ranked with the baseline model for all queries... ')
    print("\nThe total number of the documents for each query is: ", len(retrieved_docs_total[1]))

    # Evaluate the baseline model with mean of precisions at r=50 and MRR
    # Mean of Precisions for the top 50 documents: 0.097
    print("\nMean of Precisions for the top 50 documents retrieved with the baseline model: ", articles.precisions_mean(queries, answers, retrieved_docs_total))
    # Mean of Precisions for all documents:
    print("\nMean of Precisions for all documents retrieved with the baseline model: ", articles.precisions_mean(queries, answers, retrieved_docs_total, r=8749))

    # Mean reciprocal rank with the top 50 documents: 0.5913
    print("\nMRR for the top 50 documents retrieved with the baseline model: ", articles.mean_reciprocal_rank(answers, retrieved_docs_50))
    # Mean reciprocal rank when taking all documents: 0.592
    print("\nMRR for all documents retrieved with the baseline model: ", articles.mean_reciprocal_rank(answers, retrieved_docs_total))
