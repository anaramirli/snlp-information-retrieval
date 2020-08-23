#Authors
- Anar Amirli, 2581604, s8aramir@stud.uni-saarland.de
- Ga Yeon Roß, 2568941, gayeonross1@gmail.com 

# snlp
In this project, the task is to develop and evaluate a two-stage information retrieval model that given a query returns the n most relevant documents and then ranks the sentences within the documents. For the first part, we implement a baseline document retriever with tf-idf features. In the second part we improve over the baseline of the document retriever with an advanced approach of your choice. The third part extends the model to return the ranked sentences. The answer to the query should be found in one of the top-ranked sentences. 


#About the codes 

- baseline_doc_retriever.py

	(Executes task 1 of the assignment)

	This code ranks the documents with the baseline model using tf-idf scores for each query. 
	It calculates the mean of precisions and the mean reciprocal rank for the top 50 documents retrieved by the baseline model.

	The whole process takes around 6 minutes. 

- bm25_reranker.py 

	(Executes task 2 and 3 of the assignment)

	This code uses the Okapi BM25 model to get the top 50 documents based on the top 1000 documents retrieved by the baseline model and also to get the top 50 sentences from the top 50 documents.
	The performance of the BM25 model will be evaluated with mean of precisions and mean reciprocal rank. 

	The whole process takes around 6 minutes. 

- bm25_fine_tuning.py

	This script is extra, not directly associated with the tasks given in the assignment. This code is used to fine-tune BM25 parameters on both corpuses. Finding best values for parameters, we later use them in bm25_reranker.py when we execute assignment tasks. Beside finding the best result along with corresponding params, furthermore it also plots all the results over different params.

	The whole process takes around 30 minutes. 

#How to run the code, requirements

To run the codes you need python version 3 and following libraries:

bs4; nltk; math; collections; operator; numpy; re; rank_bm25

- For the 1. task run

	python baseline_doc_retriever.py -d path/to/documents -q path/to/queries -a path/to/answers

- For the 2. task run 

	python bm25_reranker.py  -d path/to/documents -q path/to/queries -a path/to/answers

- For the extra script for param search run
        
	python bm25_fine_tuning.py -d path/to/documents -q path/to/queries -a path/to/answers
