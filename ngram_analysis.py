import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

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
    ">=5-gram": (5, 6)
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
