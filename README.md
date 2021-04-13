This project was a part of SS20 Statistical Natural Language Processing course at Universität des Saarlandes. It has been implemented by:
- Ga Yeon Roß, gayeonross1@gmail.com
- Anar Amirli, anaramirli@gmail.com

# baseline document retrieval
In this project, we did the comparison of baseline document retrieval model with a more advanced re-ranker method, Okapi BM25. Here, we not only compared these two document retrieval methods, but also examined how the result of the BM25 model can be improved and how it’s performance changes upon different corpus types. More specifically, we developed and evaluated a two-stage information retrieval model that given a query returns the n most relevant documents and then ranks the sentences within the documents. Initially, we implement a baseline document retriever with tf-idf features. Based on the model's score weights, the relevance of the document for the query is determined base on their cosine similarity with the query. Later, we improve over the baseline of the document retriever with a refined BM25 approach using Okapi BM25. Contrary to the TF-IDF approach which rewards term frequency and analyzes document frequency, the BM25 approach additionally accounts for term frequency saturation and document field-length normalization. Taking advantage of auxiliary parameters which ensure the aforementioned characteristics of the BM25, we later improved model performance by fine-tuning them in order to find the most relevant top-ranked sentences. Parameter search is done with a grid-search approach on the parameter space, using the precisions mean function as accuracy metrics. For evaluation of the performance of final models, we used the mean of precisions and mean reciprocal rank evaluation functions. Consequently, we found that BM25 outperforms the baseline model. Test results also concluded that BM24 performs better on a corpus with the n most relevant documents than the corpus which consists of top n sentences within the relevant documents. Moreover conducting fine-tuning, we found out that the BM25 model performs better on the corpora we tested with small term-frequency saturation and field-length normalization values.

</br>

**Performance overview:**

| Evaluation function | Baseline (top 50 documents) | BM25 (top 50 documents) | BM25 (top 50 sentences)
| :--- | :---: | :---: | :---: 
| Mean of precisions | 0.097 | 0.122 | 0.083
| Mean reciprocal rank (MRR) | 0.591 | 0.716 | 0.548

</br>

The BM25 function has two important parameters, namely k1 and b. As it described more in detail in the discussion part, we know that the parameter k1 controls how quickly an increase in term frequency results in term-frequency saturation. Lower values result in quicker saturation, and higher values in slower saturation. The parameter b controls how much effect field-length normalization should have. A value of 0.0 disables normalization completely, and a value of 1.0 normalizes fully. Based on this intuition, we decided to fine-tune these parameters. Above results for both BM25 models are the best results achieved by doing parameter tuning. Fine-tuning is done on the param space K1={0.05, 0.2, 0.5, 0.75, 1.5, 2.25, 2, 3, 4} and B = {0.05, 01, 0.25, 0.5, 0.75, 1} and mean precision is used as an evaluation metric. We fine-tuned parameters on both top 50 documents and top 50 sentences corpuses, then later did the ranking score calculation with best parameters, where best parameter found were k1=0.75 , b=0.05 and  k1=0.5 , b=0.05 for 50 top documents and top 50 sentences corpuses respectively. More detailed overview of parameters search can be found in following figures.

</br>
</br>

![](https://github.com/anaramirli/snlp/blob/master/assets/result1.png)
</br>
![](https://github.com/anaramirli/snlp/blob/master/assets/result2.png)

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

- For the 1. task run:

    ```
    python baseline_doc_retriever.py -d path/to/documents -q path/to/queries -a path/to/answers
    ```
- For the 2. task run:

    ```
    python bm25_reranker.py  -d path/to/documents -q path/to/queries -a path/to/answers
    ```

- For the extra script for param search run:

    ```
    python bm25_fine_tuning.py -d path/to/documents -q path/to/queries -a path/to/answers
    ```
    
- Check requirements (to run the codes you need python version 3.6+):

    ```
    conda create -n my_env python=3.6
    conda activate my_env
    (my_env)$ pip install -r requirements.txt 
    ```
