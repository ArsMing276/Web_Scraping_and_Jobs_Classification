# Web Scraping on Monster.com and Job Classification

In this project, we crawled the largest job board in US -- Monster with urlib2, bs4 and so on, then we applied some classification algorithms to categorize the jobs we just crawled accoring to job title and job description, hopefully we can get a better categorization of jobs than what Monster did. Monster had 79 categories, but some categories are vague and some are overlapping, which is what we want to improve.

1. Crawled approximately 120 thousands raw job titles within 79 categories on Monster.com as our training data, Performed
data cleaning and Created word frequency table using urllib2 beautiful soup, etc.
2. Adopted PCA, K-means methods to our data for dimensionality reduction.
3. Applied three machine learning methods–KNN, SVM, Random Forest and Bagging on our crawled data to build the Classifiers in R.
4. Chose the best classifier according to the prediction accuracy based on nearly 40 thousands test data.
5. Created a variable called confidence score to represent the probability that an observation is classified into each category.
6. Performed hypothesis test on the goodness of classification with confidence score to determine whether we should reject the classifier’s choice of category using Pearson chi-square test and Yates's correction for continuity.
