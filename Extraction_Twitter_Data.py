import csv
import snscrape.modules.twitter as sntwitter

tweets = []
max_tweets = 10000  # Specify the desired number of tweets

# Search query and other parameters
query = 'Extraction 2'
since_date = '2023-06-16'

# Iterate over search results
for tweet in sntwitter.TwitterSearchScraper(f'{query} since:{since_date} lang:en').get_items():
    tweets.append({
        'ids': tweet.id,  # Unique ID of the tweet
        'date': tweet.date.strftime('%Y-%m-%d'),  # Date of the tweet
        'flag': query if query else 'NO QUERY',  # Query or 'NO QUERY'
        'user': tweet.user.username,  # Name of the user that tweeted
        'text': tweet.content,  # Text of the tweet
        # Add other desired tweet attributes
    })
    if len(tweets) >= max_tweets:
        break
    
# Define CSV file path
csv_file = 'tweets_extraction 2.csv'

# Define field names for the CSV file
field_names = ['ids', 'date', 'flag', 'user', 'text']

# Write tweets to CSV file
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(tweets)

print(f'Tweets saved to {csv_file}')
