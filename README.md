# Web Scraping on Monster.com and Job Classification

In this project, we crawled the largest job board in US -- Monster with urlib2, bs4 and so on, then we applied some classification algorithms to categorize the jobs we just crawled accoring to job title and job description, hopefully we can get a better categorization of jobs than what Monster did. Monster had 59 categories, but some categories are vague and some are overlapping, which is what we want to improve.

1. Crawled approximately 120 thousands raw job titles within 59 categories on Monster.com as our training data, Performed
data cleaning and Created word frequency table using urllib2 beautiful soup, etc.
2. Adopted TF-IDF, word2vec algorithms to our data for dimentionality reduction.
3. Applied three machine learning methods–SVM, Random Forest and Naive Bayes on our crawled data to build the Classifiers in R.
4. Evaluated prediction accuracy based on nearly 40 thousands test data.
