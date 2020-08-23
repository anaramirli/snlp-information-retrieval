from baseline_doc_retriever import *
from bm25_reranker import *
import nltk
import optparse

def plot_result_matrix(rm,
                          x_names,
                          y_names,
                          x_label = 'k1 params',
                          y_label = 'b params',
                          title='some result matrix',
                          cmap=None,
                          normalize=False):
    """
    this is the modified verison of confusion matrix from sklearn that  make a nice plot given a result matrix (rm)

    Arguments
    ---------
    rm:           result matrix 
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_result_matrix(rm           = rm,                     # result matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(rm) / float(np.sum(rm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(rm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if x_names is not None and y_names is not None:
        x_tick_marks = np.arange(len(x_names))
        y_tick_marks = np.arange(len(y_names))
        plt.xticks(x_tick_marks, x_names, rotation=45)
        plt.yticks(y_tick_marks, y_names)

    if normalize:
        rm = rm.astype('float') / rm.sum(axis=1)[:, np.newaxis]


    thresh = rm.max() / 1.5 if normalize else rm.max() / 2
    for i, j in itertools.product(range(rm.shape[0]), range(rm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(rm[i, j]),
                     horizontalalignment="center",
                     color="white" if rm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:0.4f}".format(rm[i, j]),
                     horizontalalignment="center",
                     color="white" if rm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()
    
    
def rerank_bm25_fine_tuning(irmodel, bm25model,
                            answers, queries, tokenized_queries, documents, tokenized_docs, K, B, 
                            title='some result matrix'):
    
    """
    used to fine-tune params for the corpus returned by BM25 using preceision mean as the evaluation metrics. 
    
    :param irmodel(object): baseline model
    :param bm25model(object): ranking model  
    :param answers(list): list of lists with regex patterns as strings
    :param queries(list): contains strings of queries
    :param queries(list): contains list of tokenized queries -> [['what', 'does', 'the', 'peugeot', 'company', 'manufacture'], ...]
    :param documents(list): raw documents as strings for each query -> [['A Malaysian English', '...'], ...]
    :param tokenized_docs(list): tokenized documents for each query -> [[['a', 'malaysian', 'english', '...'], ...]
    :param K(list): param space for k1 -> [0, 0.2, 0.5, 1.5, 2, ...]
    :param B(list): param space for b -> [0, 0.1, 0.5, ... 1]
    :param title(str): title of the plot
    
    :output print best result along with corresponding params. Furthermore plot all the results over different params.
    :return best_k1, best_b(floats): best params found
    """
    
    # some accumulators for the search 
    all_results = list() # stores all the results
    best_k1 = None
    best_b  = None
    best_result = -1

    # iteratre over parms-sapce
    for b in B:
        tmp_results = list() # store each result over given b
        for k1 in K:
            top_50_raw = bm25model.rerank_bm25(tokenized_queries, documents, tokenized_docs, k1, b)
            res = articles.precisions_mean(queries, answers, top_50_raw)
            tmp_results.append(res)
            
            if res > best_result:
                best_result = res
                best_b  = b
                best_k1 = k1

        all_results.append(np.array(tmp_results).flatten())

    print('\nBest parameters found: k1={} , b={}'.format(best_k1, best_b))
    print('\nMean of Precisions: ', best_result)

    plot_result_matrix(rm = np.array(all_results),
                  x_names = K,
                  y_names = B,
                  title = title)
    
    return best_k1, best_b




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
    
    # use BASELINE model and get the top 1000 documents for each query
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

    # START PARAM SEARCH
    # define search space for saturation params and field-length normalization for fine-tuning
    K = [0.05, 0.2, 0.5, 0.75, 1.5, 2.25, 2, 3, 4] 
    B = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
    
    print("Start tine-tuning for the top 50 documents ranked with BM25....")
    k1, b = rerank_bm25_fine_tuning(articles,
                        bm25model,
                        answers, 
                        queries, 
                        queries_tokenized, 
                        top_1000_raw, 
                        top_1000_tokenized, 
                        K, B, 
                        title='Result of top 50 sentences ranked with BM25 upon different k1 and b params')
    
    top_50_raw = bm25model.rerank_bm25(queries_tokenized, top_1000_raw, top_1000_tokenized, k1, b)
    
    # Split the top 50 documents into sentences.
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

    # Treat the sentences like documents to rank them and return the top 50 sentences ranked with BM25.
    print("Start tine-tuning for the top 50 sentences ranked with BM25....")
    k1, b = rerank_bm25_fine_tuning(articles,
                        bm25model,
                        answers, 
                        queries, 
                        queries_tokenized, 
                        top_50_doc2sent_raw, 
                        top_50_doc2sent_tokenized, 
                        K, B, 
                        title='Result of top 50 sentences ranked with BM25 upon different k1 and b params')