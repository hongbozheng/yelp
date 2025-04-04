import pandas as pd
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Ignore warning messages for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Define the path to the Illinois-specific Yelp dataset
data_path = "yelp_illinois/"  # Modify this path to your dataset directory

# ---------------------------
# 1. Load Review Data
# ---------------------------

# Load the reviews dataset filtered for Illinois businesses
review_df = pd.read_json(f"{data_path}/review_illinois.json", lines=True)


# ---------------------------
# 2. N-gram Analysis
# ---------------------------

def extract_phrases(df, column, ngram_range=(1, 1), top_n=30):
    """
    Extracts top N phrases from the specified text column using TF-IDF vectorization.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the review data.
    - column (str): The name of the text column to analyze.
    - ngram_range (tuple): The range of n-grams to consider (e.g., (1, 1) for unigrams, (2, 2) for bigrams).
    - top_n (int): The maximum number of phrases to extract.

    Returns:
    - phrases (list): A list of top phrases sorted by their TF-IDF score.
    """
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range, max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform(df[column].dropna())
    phrases = vectorizer.get_feature_names_out()
    return phrases


# Split the reviews into high and low rating categories
high_rating_reviews = review_df[review_df['stars'] >= 4]
low_rating_reviews = review_df[review_df['stars'] <= 2]

# Define the n-gram ranges to analyze
ngram_ranges = {
    "1-gram": (1, 1),
    "2-gram": (2, 2),
    "3-gram": (3, 3),
    ">=5-gram": (5, 5)
}

# Dictionaries to store the results
high_phrases = {}
low_phrases = {}

# Extract top phrases for high and low rating reviews
for key, ngram_range in ngram_ranges.items():
    high_phrases[key] = extract_phrases(high_rating_reviews, 'text', ngram_range=ngram_range, top_n=30)
    low_phrases[key] = extract_phrases(low_rating_reviews, 'text', ngram_range=ngram_range, top_n=30)

# ---------------------------
# 3. Display Results
# ---------------------------

print("\nN-gram Analysis Results:")

for key in ngram_ranges.keys():
    print(f"\nHigh Rating Phrases ({key}):")
    for phrase in high_phrases[key]:
        print(f"- {phrase}")

    print(f"\nLow Rating Phrases ({key}):")
    for phrase in low_phrases[key]:
        print(f"- {phrase}")


def generate_wordcloud(df, column, ngram_range=(2, 2), top_n=50):
    """
    Generate a word cloud from TF-IDF scores for a given n-gram range.

    Parameters:
      df (pd.DataFrame): DataFrame containing reviews.
      column (str): Text column to analyze.
      ngram_range (tuple): ngram range for analysis, e.g., (2,2) for bigrams.
      top_n (int): Maximum number of phrases to include.

    Returns:
      WordCloud object.
    """
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(df[column].dropna())
    phrases = vectorizer.get_feature_names_out()
    # Sum TF-IDF scores for each phrase across all documents
    scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
    phrase_scores = dict(zip(phrases, scores))
    # Keep only the top_n phrases by score
    top_phrases = dict(sorted(phrase_scores.items(), key=lambda item: item[1], reverse=True)[:top_n])

    # Create and return the word cloud
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_phrases)
    return wc


# ---------------------------
# 2. Generate Word Clouds for 2-gram and 3-gram (High Rating Reviews)
# ---------------------------
# High Rating 2-gram Word Cloud
wc_2gram = generate_wordcloud(high_rating_reviews, 'text', ngram_range=(2, 2), top_n=30)
plt.figure(figsize=(12, 6))
plt.imshow(wc_2gram, interpolation='bilinear')
plt.axis('off')
plt.title('High Rating Reviews - 2-gram Word Cloud')
plt.show()

# High Rating 3-gram Word Cloud
# wc_3gram = generate_wordcloud(high_rating_reviews, 'text', ngram_range=(3, 3), top_n=30)
# plt.figure(figsize=(12, 6))
# plt.imshow(wc_3gram, interpolation='bilinear')
# plt.axis('off')
# plt.title('High Rating Reviews - 3-gram Word Cloud')
# plt.show()

wc_2gram = generate_wordcloud(low_rating_reviews, 'text', ngram_range=(2, 2), top_n=30)
plt.figure(figsize=(12, 6))
plt.imshow(wc_2gram, interpolation='bilinear')
plt.axis('off')
plt.title('Low Rating Reviews - 2-gram Word Cloud')
plt.show()

# High Rating 3-gram Word Cloud
# wc_3gram = generate_wordcloud(low_rating_reviews, 'text', ngram_range=(3, 3), top_n=30)
# plt.figure(figsize=(12, 6))
# plt.imshow(wc_3gram, interpolation='bilinear')
# plt.axis('off')
# plt.title('Low Rating Reviews - 3-gram Word Cloud')
# plt.show()
