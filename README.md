# Russian News classification
Multiclass classification project. Predicting news topics.

## Data
The data for the project was scraped from the news website ria.ru and parsed independetly. 
The final dataframe contains 42 210 rows (each row is a news article) covering 9 topics (9 classes): 
* science
* society
* incidents
* culture
* religion
* politics
* economy
* world
* defense_army

For each news we have these features:

| Feature Name | Data Type          |
| -------------- | --------------------- |
| url                | string                 |
| title              | string                 |
| content         | string                 |
| datetime     | datetime |
| views        | integer
| tags         | string
| topic        | string
| year | integer|
|month | integer|
|weekday | string|

The target variable is **topic**.  

The Topics Distribution is presented below:

![](https://github.com/ChernayaAnastasia/Screenshots/blob/master/topic_distribution.png)

The Text Statistics:


|                 | Train                         | Test    |  
|----------------------|-----------------------------------|-------|
| Number of Docs in the Corpus               | 33 768                       | 8 442 | 
| Number of Tokens               | 7 762 439                       | 1 915 384	  | 
| Size of Vocabulary             |  273 386	                     |  138 162           |
| Number of lemmas                | 52 55 431                             | 1 296 314  |
| Size of lemmatized vocabulary                | 112 078	                             | 57 404  | 

## Models
Logistic Regression with Tfidf Vectorizer (lemmatization + deleting stopwords)

Catboost Classifier (lemmatized corpus, built-in Catboost text featurizing)

BERT Transformer Classifier (rubert-tiny2)

## Results
| Model                | Vectorizer                         | F1    | Predicting Time |
|----------------------|-----------------------------------|-------|------------------|
| LogisticRegression               | TFIDF                       | 0.881  | 2.648436           |
| LogisticRegression               | Word2Vec with Mean Pooling                       | 0.859	  | 3.920810            |
| CatBoostClassifier             |  CatBoostClassifier with lemmatized texts                     | 0.875	  | 0.837941            |
| BERT Classifier                 | rubert-tiny2                               | 0.876  | 17.861081             |


## Author
**Chernaya Anastasia** - [Telegram](https://t.me/ChernayaAnastasia), [GitHub](https://github.com/ChernayaAnastasia)

## Links
The parser file - [open in Colab](https://drive.google.com/file/d/1Qu79WtTEPSRi6wqVoIlGDHW4uTCHhBaw/view?usp=sharing)

The EDA file - [open in Colab](https://drive.google.com/file/d/1ssvoLTkO74KucYgIZt9bsBe6SJUmLws5/view?usp=sharing), [open in nbviewer](https://nbviewer.org/github/ChernayaAnastasia/News-classification/blob/main/eda_ria_news.ipynb)

## License
This project is licensed under the MIT license. For more information, see the [LICENSE](https://github.com/ChernayaAnastasia/News-classification/blob/main/LICENSE) file.

