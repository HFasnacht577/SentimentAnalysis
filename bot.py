import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression



def train_log():

    df = pd.read_csv("IMDB Dataset.csv")
    df = df.dropna()

    # Split the dataframe into text and label components
    reviews = df['review']
    sentiments = df['sentiment'].apply(lambda x:1 if x=="positive" else 0)

    # Create a TfidfVectorizer instance for the text data
    vectorizer = TfidfVectorizer()

    # Fit and transform the reviews data into a TF-IDF matrix
    review_vecotrs = vectorizer.fit_transform(reviews)


    model = LogisticRegression() #create logistic regression model

    model.fit(review_vecotrs,sentiments) #train model

    return model, vectorizer



