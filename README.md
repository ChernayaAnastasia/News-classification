# News classification
Text multiclass classification project. Predicting news topics.

## Data
The data for the project was scraped from the news website ria.ru and parsed independetly. 
The final dataframe contains 23 985 rows (each row is a news article) covering 7 topics: 
* science
* society
* incidents
* culture
* politics
* economy
* world

For each news we have these features:

| Feature Name | Data Type          |
| -------------- | --------------------- |
| url                | string                 |
| title              | string                 |
| content         | string                 |
| datetime     | datetime |
| year | integer|
|month | integer|
|weekday | string|
|len_title| integer|
|len_content| integer
|topic | string

The target variable is **topic**

The dataset corpus includes **3 971 976** unlemmatized tokens stopwords excluded. 

The topics distribution is presented below:

![](https://github.com/ChernayaAnastasia/Screenshots/blob/master/topics_distribution.png)

## Models
Under development

## Results
Under development

## Author
**Chernaya Anastasia** - [Telegram](https://t.me/ChernayaAnastasia), [GitHub](https://github.com/ChernayaAnastasia)

## Links
The parser file - [open in Colab](https://drive.google.com/file/d/1Qu79WtTEPSRi6wqVoIlGDHW4uTCHhBaw/view?usp=sharing)

The EDA file - [open in Colab](https://drive.google.com/file/d/1ssvoLTkO74KucYgIZt9bsBe6SJUmLws5/view?usp=sharing), [open in nbviewer](https://nbviewer.org/github/ChernayaAnastasia/News-classification/blob/main/eda_ria_news.ipynb)

## License
This project is licensed under the MIT license. For more information, see the [LICENSE](https://github.com/ChernayaAnastasia/News-classification/blob/main/LICENSE) file.

