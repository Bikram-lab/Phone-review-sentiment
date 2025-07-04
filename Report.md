**Sentiment Classification & Clustering for cell phone reviews**

**Table of Contents**

1.  Abstract
2.  Project Background
3.  Analysis Approach
    -   Dataset
    -   Data Processing
    -   Variables
    -   Text Mining and Exploratory Data Analysis
    -   Prediction Modeling and Evaluation
4.  Results and Discussion
    -   Descriptive Statistics of Text Corpus
    -   Text Mining of Reviews
    -   Association Rules Mining of Reviews
    -   Cluster Analysis of Reviews
    -   Tree-based and Logistic-based Prediction Modeling
    -   Discriminant Analysis Based Modeling
    -   Support Vector Machines Based Modeling
5.  Comparative Analysis of Prediction Models
6.  Conclusions
7.  References
8.  Appendix

**Abstract**  
This project examines sentiment classification and clustering of mobile phone reviews using text mining and predictive modeling methods. Drawing from a large corpus of reviews, the analysis aimed to uncover hidden associations, form document clusters, and build accurate sentiment prediction models. Techniques included n-gram generation, sentiment scoring, association rule mining, unsupervised clustering (K-means, DBSCAN, and hierarchical), and predictive modeling (logistic regression, SVM, decision trees, XGBoost). The project demonstrates the effectiveness of ensemble and linear classifiers, identifies data distribution limitations impacting clustering performance, and concludes with practical recommendations for implementation in real-time review monitoring systems.

**Project Background**  
Understanding consumer sentiment from product reviews is critical for businesses aiming to improve product features and customer satisfaction. With the exponential growth in user-generated content, leveraging unstructured review data for actionable insights has become increasingly important. This project explores sentiment classification and clustering of cell phone reviews using machine learning techniques.

Three research questions were addressed:

1.  Can review sentiment be reliably predicted using word vectors, sentiment scores, and metadata?
2.  Are there distinguishable clusters of reviews based on sentiment and term associations?
3.  Which modeling techniques best capture sentiment variance in this dataset?

These questions are pertinent for digital marketers, product analysts, and sentiment-aware recommender systems, enabling targeted strategies and automated feedback categorization. The dataset was sourced from Kaggle’s 14 Million Cell Phone Reviews corpus which includes over 150,000 reviews. A representative sample of 15,000 was selected based on product stratification. Modules 1–5 contributed sequentially: sentiment analysis and text mining (Module 1), tree and logistic models (Module 2), association rules and discriminant analysis (Module 3), clustering (Module 4), and SVM modeling (Module 5).

**Analysis Approach**

1.  **Dataset**

The reviews dataset covers feedback across hundreds of mobile phone models. After stratified sampling, 15,000 reviews were selected, ensuring representation across positive, negative, and neutral classes. The raw dataset included review text, timestamps, and product IDs.

2.  **Data Processing**

Data preprocessing included:

-   Text normalization (lowercasing, punctuation removal)
-   Stopword removal
-   TF-IDF vectorization (top 200 terms)
-   Sentiment scoring using AFINN, Bing, and NRC lexicons
-   Metadata derivation: review length, word count, punctuation ratios
-   Missing value imputation and scaling for clustering
3.  **Variables**

Response variable: Sentiment class (positive, neutral, negative) inferred from review ratings  
Predictor variables:

-   TF-IDF word vectors (200 features)
-   Sentiment scores (Bing, AFINN, NRC)
-   Metadata (review length, punctuation use, caps ratio)
4.  **Text Mining and Exploratory Data Analysis**

Sentiment score distributions (Figure 2), n-gram frequencies, and sentiment polarity were examined. Score by sentiment was visualized (Figure 1). Association rules were mined for positive/negative sentiments (Tables 3–4).

5.  **Prediction Modeling and Evaluation**

The following models were developed and evaluated:

-   Logistic Regression
-   XGBoost
-   Random Forest
-   Linear and Regularized Discriminant Analysis
-   Support Vector Machines (linear and RBF)

Each model was trained using stratified 70:30 train-test split. Metrics included accuracy, precision, recall, and F1-score. Key packages included caret, xgboost, randomForest, MASS, klaR, e1071, and text2vec.

**Results and Discussion**

1.  **Descriptive Statistics of Text Corpus**

Table 1 shows average review length. Sentiment distribution is shown in Figure 2. Word count distribution (Figure 4) revealed skewness toward medium-length reviews.

2.  **Text Mining of Reviews**

We applied unigrams and bigrams to identify common patterns. Sentiment polarity was explored using lexicons. Positive sentiment was dominant across reviews.

3.  **Association Rules Mining of Reviews**

Strong associations were found for negative reviews containing words like “battery” and “poor” (Table 3), while positive reviews often referenced “value” and “great” (Table 4).

4.  **Cluster Analysis of Reviews**

K-means, hierarchical, and DBSCAN clustering were applied using multiple distance metrics (Table 8). Cluster visualizations (Figure 7, Figure 9), DBSCAN summary (Table 11), and cluster-sentiment purity (Table 13) were included. K=2 was optimal for K-means based on silhouette score (Table 9). DBSCAN found 8 clusters using Euclidean distance with 5.8% noise.

5.  **Tree-based and Logistic-based Prediction Modeling**

Logistic regression and XGBoost outperformed random forests (Table 14). Feature importance from XGBoost is shown in Figure 6. Confusion matrices are detailed in Figure 11a and 11b.

6.  **Discriminant Analysis Based Modeling**

Linear Discriminant Analysis achieved moderate accuracy (57.6%), while Regularized Discriminant Analysis underperformed (50.3%).

7.  **Support Vector Machines Based Modeling**

Linear SVM achieved 66.8% accuracy, outperforming RBF kernel (64.9%). This reflects the dataset’s favorability toward linear separation (Table 14).

**Comparative Analysis of Prediction Models**  
Figure 5 and ROC curves show that Logistic Regression and XGBoost consistently outperformed other models. SVM models followed closely, while discriminant models performed poorly. K-means and DBSCAN revealed limited clustering potential, likely due to imbalanced sentiment distribution. Table 14 and the model summary table list all comparative metrics.

**Conclusions  
**This project demonstrated how natural language processing and supervised learning can classify review sentiment with high accuracy. Logistic regression and XGBoost were most effective. Although clustering revealed limited cohesion, association rules and lexicons helped characterize sentiment-rich reviews. The results highlight the value of ensemble methods and the challenge of unsupervised techniques on sentiment-imbalanced data. We recommend implementing the logistic regression model for real-time sentiment classification due to its simplicity and interpretability.

**References**

-   Kassambara (2018, November 11). Discriminant analysis essentials in R. <https://www.sthda.com/english/articles/36-classification-methods-essentials/146-discriminant-analysis-essentials-in-r/>
-   Gulati J. (2024, December 2024). How to perform sentiment analysis in R. <https://www.statology.org/how-to-perform-sentiment-analysis-r/>
-   GeeksforGeeks. (2025, May 7). TF-IDF for bigrams & trigrams. <https://www.geeksforgeeks.org/tf-idf-for-bigrams-trigrams/>
-   Prabhakaran S. (n.d.). Association Mining (Market Basket Analysis) <http://r-statistics.co/Association-Mining-With-R.html>
-   Hahsler, M., Piekenbrock, M., & Doran, D. (2019). dbscan: Density Based Clustering of Applications with Noise (DBSCAN) and Related Algorithms. (Version 1.1-5) [Computer software]. <https://CRAN.R-project.org/package=dbscan>
-   R Core Team. (2024). R: A language and environment for statistical computing [Computer software]. R Foundation for Statistical Computing. <https://www.R-project.org/>
-   Meyer, D., & Buchta, C. (2021). proxy: Distance and Similarity Measures (Version 0.4-27) [Computer software]. <https://CRAN.R-project.org/package=proxy>
-   UC Business Analytics R Programming Guide. (n.d.). *K-means Cluster Analysis*. <https://uc-r.github.io/kmeans_clustering>
-   UC Business Analytics R Programming Guide. (n.d. ). *Hierarchical Clustering*. <https://uc-r.github.io/hc_clustering>
-   DataCamp. (2024, September 29). *A Guide to the DBSCAN Clustering Algorithm*. <https://www.datacamp.com/tutorial/dbscan-clustering-algorithm>

**Appendix**

-   Table 1: Dataset Descriptive Statistics  
    ![A screenshot of a report AI-generated content may be incorrect.](media/3c97cd67a7ea09dc2e7061d80e206079.png)
-   Figure 1: Score Distribution

    ![A graph of a bar AI-generated content may be incorrect.](media/15be696ce2987897509fa98b6f94d17f.png)

-   Figure 2: Sentiment Distribution  
    ![A graph of a number of different colored squares AI-generated content may be incorrect.](media/7757a80b372c5d435c427d00138550a5.png)
-   Figure 3: Score by Sentiment

    ![A graph with a bar chart and a graph with a bar chart and a graph with a graph AI-generated content may be incorrect.](media/a04847055fa6c94dff5814bb4c8a07d7.png)

-   Figure 4: Review Length Distribution (Characters)

    ![A graph of a number of characters AI-generated content may be incorrect.](media/a9f72077b48a2747b43c28c0f53cca83.png)

-   Table 2: Correlation between Review Characteristics and Sentiment

    ![A white background with black text AI-generated content may be incorrect.](media/08184392afd18efe9603a38dd51099ea.png)

-   Table 3: Top Association Rules for Negative Sentiment

    ![](media/c60052ea4fed85699a8012a7005489d7.png)

-   Table 4: Top Association Rules for Positive Sentiment

    ![A screenshot of a computer AI-generated content may be incorrect.](media/569717624e48c18e91e04d057a35526a.png)

-   Figure 6: Feature Importance (XGBoost)  
    ![A graph with blue bars AI-generated content may be incorrect.](media/49c62bde0fc79cdd4afbdc9a1227697d.png)
-   Model Accuracy summary table

    ![A screenshot of a computer AI-generated content may be incorrect.](media/3855af512db96ccfbc0638e1b04ca07b.png)

-   Figure 5: Model Performance Comparison

    ![A graph of a graph showing different colored bars AI-generated content may be incorrect.](media/67bc2d5567d99ed0884b1791fcdeafcb.png)

-   ROC Curves: All Models (Positive Class)

    ![A graph with numbers and lines AI-generated content may be incorrect.](media/bcc4c55b05dfa17fb54c6549878f497d.png)

-   Table 14: Model Performance Metrics

    ![A screenshot of a computer AI-generated content may be incorrect.](media/299f8be6ce9160abbc8ffa0d4c954d8d.png)

-   Figure 11a: Confusion Matrics(Part1)

    ![A screenshot of a graph AI-generated content may be incorrect.](media/d36f270d64a2c338936269af2dc7fb8c.png)

-   Figure 11b: Confusion Matrics(Part2)

    ![A screenshot of a graph AI-generated content may be incorrect.](media/011621c3e22aa57ee0764443bbfef14b.png)

-   Table 8: Distance Metrics Comparison

    ![A white text with black text AI-generated content may be incorrect.](media/8dd97fbc3dde6195be01a880915e30bc.png)

-   Distribution for Distance

    ![A graph of different sizes and distances AI-generated content may be incorrect.](media/3f2da4d51c02f45f296fcf4a0d647c37.png)

-   Figure 7: K-means Clustering vs True Labels

    ![A screenshot of a graph AI-generated content may be incorrect.](media/ba1abced027ca5808b6f020f672be468.png)

-   Table 11: DBSCAN Clustering Results Summary

    ![A close-up of a white background AI-generated content may be incorrect.](media/ac52b9f2cdea5ad2a846316829914f83.png)

-   Figure 9: DBSCAN Clustering Results Analysis

    ![A screenshot of a graph AI-generated content may be incorrect.](media/bf2e8a175bf2b96f473a68570e798951.png)

-   Table 10: Hierarchical Clustering Methods Comparison

    ![A white paper with black text AI-generated content may be incorrect.](media/7ee8c5fa6e028385e5eb6c4a54484352.png)

-   Ward Linnkage(Euclidean) Dendogram for Hierarchical Clustering Methods

    ![A diagram of a graph AI-generated content may be incorrect.](media/6e453c1f12dda7840e782e28cd17808a.png)

-   Table 13: Cluster-Sentiment Relationship Analysis

    ![A screenshot of a computer AI-generated content may be incorrect.](media/fbaf1c0f894d81fe3186b83232710e1d.png)
