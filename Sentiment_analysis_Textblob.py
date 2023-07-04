import numpy as np
import pandas as pd
import datatable as dt
#reading
data = dt.fread('./cleaned_tweets_extraction.csv')
df = data.to_pandas()

from textblob import TextBlob
import matplotlib.pyplot as plt

# Define a function to perform sentiment analysis on the tokenized text using TextBlob
def get_sentiment(tokens):
    text = ' '.join(tokens)
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Apply the function to the 'tokens' column of the DataFrame
df['sentiment'] = df['tokens'].apply(get_sentiment)

# Print the DataFrame with sentiment scores
print(df)





from textblob import TextBlob

def sentiment_analysis(df):
    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity

    def getPolarity(text):
        return TextBlob(text).sentiment.polarity

    def getAnalysis(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'

    # Create two new columns 'TextBlob_Subjectivity' and 'TextBlob_Polarity'
    df['TextBlob_Subjectivity'] = df['tokens'].apply(getSubjectivity)
    df['TextBlob_Polarity'] = df['tokens'].apply(getPolarity)
    df['TextBlob_Analysis'] = df['TextBlob_Polarity'].apply(getAnalysis)

    return df

# Apply sentiment analysis using TextBlob on the 'df' DataFrame
df = sentiment_analysis(df)

import matplotlib.pyplot as plt
# Count the occurrences of each sentiment
sentiment_counts = df['TextBlob_Analysis'].value_counts()

# Plot the sentiments
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Analysis')
plt.show()

# Print the number of counts for each sentiment
for sentiment, count in sentiment_counts.items():
    print(f"{sentiment}: {count}")


