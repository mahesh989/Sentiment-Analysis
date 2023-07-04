pip install vaderSentiment

#import the necessary libraries
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import datatable as dt

#reading
data = dt.fread('./cleaned_tweets_extraction.csv')
df = data.to_pandas()

#Create an instance of the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

#Define a function to perform sentiment analysis using VADER
def get_sentiment(tokens):
    sentiment = analyzer.polarity_scores(tokens)
    compound_score = sentiment['compound']
    return compound_score

# Apply the function to the 'tokens' column of the DataFrame
df['sentiment'] = df['tokens'].apply(get_sentiment)

# Print the DataFrame with sentiment scores
print(df['sentiment'])

# Count the occurrences of each sentiment
sentiment_counts = pd.cut(df['sentiment'], bins=3, labels=['Negative', 'Neutral', 'Positive']).value_counts()

# Plot the sentiments
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Analysis')
plt.show()

# Print the number of counts for each sentiment
for sentiment, count in sentiment_counts.items():
    print(f"{sentiment}: {count}")