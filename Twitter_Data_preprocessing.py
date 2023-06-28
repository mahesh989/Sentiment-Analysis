import numpy as np
import pandas as pd
import datatable as dt

#reading
data = dt.fread('./tweets_extraction 2.csv')
df = data.to_pandas()


df.shape #gives number of rows and columns
df.info()  #provides datatype and respective data info null or not
df.head()  #gives first 5 rows by default
df.tail()

#Checking Null values
null_cols = df.columns[df.isnull().any()]
null_df = df[null_cols].isnull().sum().to_frame(name='Null Count')\
          .merge(df[null_cols].isnull().mean().mul(100).to_frame(name='Null Percent'), left_index=True, right_index=True)
null_df_sorted = null_df.sort_values(by='Null Count', ascending=False)
print(null_df_sorted)

count_duplicates = df[df.duplicated()].shape[0]
print("Number of duplicate rows:", count_duplicates)

#delete the duplicate entries
df.drop_duplicates(inplace=True)


# Lowercase the data
df['text_cleaned'] = df['text'].apply(lambda x: x.lower())
# Print the updated 'cleaned_text' column
df['text_cleaned'].tail(20)

import re
# Removing URLs
df['text_cleaned'] = df['text_cleaned'].apply(lambda x: re.sub(r'http\S+|www.\S+', '', x))
# Removing HTML tags
from bs4 import BeautifulSoup
df['text_cleaned'] = df['text_cleaned'].apply(lambda x: BeautifulSoup(x, "lxml").text)
# Print the updated 'cleaned_text' column
df['text_cleaned'].tail(20)


#convert chatwords 
chat_words_dict = {
    "imo": "in my opinion",
     "cyaa": "see you",
    "idk": "I don't know",
    "rn": "right now",
    "afaik": "as far as I know",
    "afk": "away from keyboard",
    "asap": "as soon as possible",
    "atk": "at the keyboard",
    "atm": "at the moment",
    "a3": "anytime, anywhere, anyplace",
    "bak": "back at keyboard",
    "bbl": "be back later",
    "bbs": "be back soon",
    "bfn": "bye for now",
    "b4n": "bye for now",
    "brb": "be right back",
    "brt": "be right there",
    "btw": "by the way",
    "b4": "before",
    "b4n": "bye for now",
    "cu": "see you",
    "cul8r": "see you later",
    "cya": "see you",
    "faq": "frequently asked questions",
    "fc": "fingers crossed",
    "fwiw": "for what it's worth",
    "fyi": "for your information",
    "gal": "get a life",
    "gg": "good game",
    "gn": "good night",
    "gmta": "great minds think alike",
    "gr8": "great!",
    "g9": "genius",
    "ic": "I see",
    "icq": "I seek you (also a chat program)",
    "ilu": "I love you",
    "imho": "in my honest/humble opinion",
    "imo": "in my opinion",
    "iow": "in other words",
    "irl": "in real life",
    "kiss": "keep it simple, stupid",
    "ldr": "long distance relationship",
    "lmao": "laugh my a.. off",
    "lol": "laughing out loud",
    "ltns": "long time no see",
    "l8r": "later",
    "mte": "my thoughts exactly",
    "m8": "mate",
    "nrn": "no reply necessary",
    "oic": "oh I see",
    "pita": "pain in the a..",
    "prt": "party",
    "prw": "parents are watching",
    "rofl": "rolling on the floor laughing",
    "roflol": "rolling on the floor laughing out loud",
    "rotflmao": "rolling on the floor laughing my a.. off",
    "sk8": "skate",
    "stats": "your sex and age",
    "asl": "age, sex, location",
    "thx": "thank you",
    "ttfn": "ta-ta for now!",
    "ttyl": "talk to you later",
    "u": "you",
    "u2": "you too",
    "u4e": "yours for ever",
    "wb": "welcome back",
    "wtf": "what the f...",
    "wtg": "way to go!",
    "wuf": "where are you from?",
    "w8": "wait...",
    "7k": "sick:-D laugher"
}

# Function to convert chat words to expansions
def convert_chat_words(text):
    words = text.split()
    converted_words = []
    for word in words:
        if word.lower() in chat_words_dict:
            converted_words.append(chat_words_dict[word.lower()])
        else:
            converted_words.append(word)
    converted_text = " ".join(converted_words)
    return converted_text

# Apply chat word conversion to the 'text_cleaned' column
df['text_cleaned'] = df['text_cleaned'].apply(convert_chat_words)
# Print the updated 'text_cleaned' column
print(df['text_cleaned'].tail(20))


# Removing punctuation
import string
df['text_cleaned'] = df['text_cleaned'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
# Removing numbers
df['text_cleaned'] = df['text_cleaned'].apply(lambda x: re.sub(r'\d+', '', x))
# Removing extra spaces
df['text_cleaned'] = df['text_cleaned'].apply(lambda x: ' '.join(x.split()))
# Replacing repetitions of punctuation
df['text_cleaned'] = df['text_cleaned'].apply(lambda x: re.sub(r'(\W)\1+', r'\1', x))
# Print the updated 'cleaned_text' column
df['text_cleaned'].tail(20)


import emoji
import re
# Function to convert emojis to words using emoji library mapping
def convert_emojis_to_words(text):
    converted_text = emoji.demojize(text)
    return converted_text
# Apply the function to the 'cleaned_text' column in the DataFrame
df['text_cleaned'] = df['text_cleaned'].apply(convert_emojis_to_words)
# Print the updated 'cleaned_text' column
df['text_cleaned'].tail(20)

# Removing special characters
df['text_cleaned'] = df['text_cleaned'].apply(lambda x: re.sub(r"[^\w\s]", '', x))
# Print the updated 'cleaned_text' column
df['text_cleaned'].tail(20)


# Removing contractions
import contractions
# Remove contractions from the 'text_cleaned' column
df['text_cleaned'] = df['text_cleaned'].apply(lambda x: contractions.fix(x))
# Print the updated 'cleaned_text' column
df['text_cleaned'].tail(20)

#check for contractions
from collections import Counter
# Define the contraction pattern
contraction_pattern = re.compile(r"\b\w+'\w+\b")
# Find all contractions in the text column
contractions = df['text_cleaned'].apply(lambda x: contraction_pattern.findall(x))
# Flatten the list of contractions
contractions = [item for sublist in contractions for item in sublist]
# Count the frequency of each contraction
contraction_counts = Counter(contractions)
# Sort contractions by descending count
sorted_contractions = sorted(contraction_counts.items(), key=lambda x: x[1], reverse=True)
# Print unique contractions and their counts in descending order
for contraction, count in sorted_contractions:
    print(contraction, count)
   

from langdetect import detect
# Iterate through each row
for index, row in df.iterrows():
    text = row['text_cleaned']
    try:
        lang = detect(text)
    except:
        lang = 'unknown'
    if lang != 'en':
        df.at[index, 'text_cleaned'] = ''  # Replace non-English text with an empty string

# Print the updated 'text_cleaned' column
df['text_cleaned'].tail(20)


from nltk.tokenize import word_tokenize
# Tokenization
df['tokens'] = df['text_cleaned'].apply(lambda x: word_tokenize(x))
# Print the updated 'tokens' column
df['tokens'].tail(20)

# Removing stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words])
# Print the updated 'tokens' column
df['tokens'].tail(20)


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
# Create an instance of WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# POS tag mapping dictionary
wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

# Function to perform Lemmatization on a text
def lemmatize_text(text):
    # Get the POS tags for the words
    pos_tags = nltk.pos_tag(text)
    
    # Perform Lemmatization
    lemmatized_words = []
    for word, tag in pos_tags:
        # Map the POS tag to WordNet POS tag
        pos = wordnet_map.get(tag[0].upper(), wordnet.NOUN)
        # Lemmatize the word with the appropriate POS tag
        lemmatized_word = lemmatizer.lemmatize(word, pos=pos)
        # Add the lemmatized word to the list
        lemmatized_words.append(lemmatized_word)
    
    return lemmatized_words

# Apply Lemmatization to the 'tokens' column
df['tokens'] = df['tokens'].apply(lemmatize_text)
# Print the updated 'tokens' column
print(df['tokens'].tail(20))

# Save the DataFrame as a CSV file
df.to_csv('cleaned_tweets_extraction.csv', index=False)



from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Create a list of all tokens
all_tokens = [token for tokens_list in df['tokens'] for token in tokens_list]
# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(nltk.FreqDist(all_tokens)))
# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


import nltk
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
# Calculate word frequency
freq_dist = FreqDist(all_tokens)
# Plot the most common words
plt.figure(figsize=(10, 5))
freq_dist.plot(30, cumulative=False)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()


