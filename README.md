# DSCI303 Final Project 
Project Name: Ad Identifier

Motivation:

The aim of this project was to develop a machine learning model to accurately identify online ads. We trained and tested multiple models, including Logistic Regression and Random Forest Classification, and found that the latter performed best. The use of machine learning in this context has the potential to improve the efficiency and effectiveness of ad identification, leading to a better user experience and potentially reducing instances of fraudulent or misleading advertising. However, this technology also raises ethical concerns, such as the potential for bias and the invasion of user privacy. To address these issues, it is important for researchers to continue to develop and refine their methods in a responsible and transparent manner.

Methods:

The dataset utilized for this project encodes the phrases and words of the html information of the images in an internet page. Additionally, it shows if an image URL is the same as a web page URL, and shows the geometric aspects of the image. 

To be more specific, the encoding contains the following information:

The geometric aspects of the data which are continuous.
Check if the server of the URL of the image is the same as the one of the URL to where the anchor points.
Each image contains a caption that encodes each of its words, and if the phrase appears more than M times in the data set.
The set of alternate words in the <IMG tag are encoded in the same manner as 3.
The same encode is used for the URL of the base, target, and image.

The next step was to determine which machine algorithm would perform the best with the clean and standardized dataset. Since the data is labeled, and it contains continuous and binary features, the team checked the classification report for Logistic Regression (Baseline Model), Random Forest, Mixed Naive Bayes, K Nearest Neighbors, and Support Vector Machine. Naive Bayes (NB) can be applied with Gaussian NB (continuous features) and Bernoulli NB (binary features) to then combine them to calculate the overall probability of being ad or non-ad. The probabilities of continuous NB times the ones from Bernoulli NB divided by the prior probability yields the overall probability of each. This follows the basic idea of the NB theorem. By performing this mixed NB the accuracy improved around 30%. Overall, most algorithms had high accuracies which are discussed in the results section.

Results:

After using several metrics to compare the performance of the different algorithms that we chose, we found Logistic Regression and Random Forest Classification to be the most efficient for our dataset. Throughout this project, we found that a majority of the algorithms that we tested actually performed quite well. But, the two mentioned above were the ones that performed the best when comparing the different metrics used to evaluate models.

A simple classification report showed us that this model has an accuracy of 0.98 and an average accuracy of 0.97. With this in mind, we adjusted the parameters found in this found model to have an accuracy of 0.98 and a weighted precision of 0.98. 

Conclusion:

In conclusion, for our dataset, it was crucial to understand the format in which we are processing the data and the fact that we had to use supervised learning based on that data. Additionally, we wanted our final chosen model to be able to scale with data, handle missing features, and be incremental to relearn the classifier. These two characteristics of our problem led us to have Logistic Regression and Random Forest Classification as our best-performing models. We saw these two models have the best performance. Although this was the case, we still believe that these models can be improved. The dataset we used had a very specific set of information (i.e. just the size and the url). If we could include other types of information about images or ads, then these models would be able to adapt to different types of advertisements. But, with the performance, we do believe that our models handle this problem efficiently and are able to adapt to new and different inputs that could be presented in the future. We considered using feature selection and even attempted to do so as previously mentioned, but since there are a large number of features even if the dataset is not too large, it was very computationally expensive to try and run the feature selection algorithm on all of the features. Our models did perform very well even without going through feature selection, but this is one aspect that could potentially improve performance even more if done in the future.


