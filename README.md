# Web Scraping on Monster.com and Jobs Classification

In this project, we crawled the largest job board in US -- Monster with **urlib2**, **bs4** and trained three **machine learning** models with **Scikit-Learn** to classify those job texts into 59 categories given by Monster.com. During the whole procedure, we also applied some **Natural Language Processing** techniques to help process the texts. 

> Main Idea of this Project

Even if with only 59 categories, maintaining a job board is still a daunting job given the extremely large volume of jobs posted each day. It's time consuming and requires a lot of labors. Besides, some jobs are hard to classify if not given enough resources. For example, when we see the job title as *"Network Support Offer Lead Job in Dallas 75201, Texas US"*, without enough specialty, it's hard for a worker to know whether to classify it in *Telecommunication Jobs* or *IT Jobs* category. Finally, given such a messy title, it is hard for us to query and aggregate any data across job titles. Ex. how do we compute how many supportors in Texas area? Thus, a good model is necessary here to help automate the classification process more accurately and speedily.

> Steps

1. Crawled approximately 120 thousands raw job descriptions within 59 categories on Monster.com as our training data.
2. Adopted **beautifulsoup** and **NLTK** to perform tokenization, remove stop words, html tags, and lemmatize words. 
3. (Feature Engineering Approach 1) Engineered features with **bag of words** and applied **TF-IDF** to re-weight the features.
4. (Feature Engineering Approach 2) Tranformed from bag-of-words counts into a topic space of lower dimensionality. In detail, we represented each document as a vector of probability distribution over 100 potential topics from **Latent Dirichlet Allocation** result.  
5. (Feature Engineering Approach 3) Projected each word to its vector representation with **word2vec** algorithm. Aggregated the embedding vectors of all words in a job description as our new feature space. There are two ways for the aggregation:
    - Average of Vector Embedding: We average all the embedding vectors in a given job decription to finally get a 500 dimensions vector as our features for that job.
    - Clustering on Vector Embedding: We cluster words into clusters according to the embedding vectors using **K-Means**. Then we count the frequency of each cluster that the words in a given job description fall into, use this frequency table as our features for that job. Finally, we also applied **TF-IDF** to re-weight the features.
6. Applied three machine learning methods–SVM, Random Forest and Naive Bayes on data from both approaches.
7. Evaluated and compared prediction accuracy of different approaches. We found Random Forest using vector representation features (aggregated by clustering) has the biggest prediction accuracy, it's 67.53%. This is far from perfect, but it's definitely much more accurate than classify randomly, for which the correct rate is only 1.7% (1/59)
