## Authors
- Ga Yeon Roß, gayeonross1@gmail.com
- Anar Amirli, s8aramir@stud.uni-saarland.de

# baseline document retrieva
In this project, we did the comparison of baseline document retrieval model with a more advanced re-ranker method, Okapi BM25. In this study, we not only compared these two document retrieval methods, but also examined how the result of the BM25 model can be improved and how it’s performance changes upon different corpus types. More specifically, we developed and evaluated a two-stage information retrieval model that given a query returns the n most relevant documents and then ranks the sentences within the documents. Initially, we implement a baseline document retriever with tf-idf features. Later, we improve over the baseline of the document retriever with a refined BM25 approach using Okapi BM25. Contrary to the TF-IDF approach which rewards term frequency and analyzes document frequency, the BM25 approach additionally accounts for term frequency saturation and document field-length normalization. Taking advantage of auxiliary parameters which ensure the aforementioned characteristics of the BM25, we later improved model performance by fine-tuning them in order to find the most relevant top-ranked sentences. Parameter search is done with a grid-search approach on the parameter space, using the precisions mean function as accuracy metrics. For evaluation of the performance of final models, we used the mean of precisions and mean reciprocal rank evaluation functions. Consequently, we found that BM25 outperforms the baseline model. Test results also concluded that BM24 performs better on a corpus with the n most relevant documents than the corpus which consists of top n sentences within the relevant documents. Moreover conducting fine-tuning, we found out that the BM25 model performs better on the corpora we tested with small term-frequency saturation and field-length normalization values.

## About the codes 

- baseline_doc_retriever.py
	
	This code ranks the documents with the baseline model using tf-idf scores for each query. 
	It calculates the mean of precisions and the mean reciprocal rank for the top 50 documents retrieved by the baseline model.

- bm25_reranker.py 

	This code uses the Okapi BM25 model to get the top 50 documents based on the top 1000 documents retrieved by the baseline model and also to get the top 50 sentences from the top 50 documents.
	The performance of the BM25 model will be evaluated with mean of precisions and mean reciprocal rank. 

- bm25_fine_tuning.py

	This code is used to fine-tune BM25 parameters on both corpuses. Finding best values for parameters, we later use them in bm25_reranker.py when we execute assignment tasks. Beside finding the best result along with corresponding params, furthermore it also plots all the results over different params.


## How to run the code, requirements

- For the 1. task run

    	```
	python baseline_doc_retriever.py -d path/to/documents -q path/to/queries -a path/to/answers
    	```
- For the 2. task run 

    	```
	python bm25_reranker.py  -d path/to/documents -q path/to/queries -a path/to/answers
    	```

- For the extra script for param search run
    	```
	python bm25_fine_tuning.py -d path/to/documents -q path/to/queries -a path/to/answers
    	```
- Check requirements, to run the codes you need python version 3.6+:
	'''
	conda create -n my_env python=3.6
	conda activate my_env
	(my_env)$ pip install -r requirements.txt 
	'''
