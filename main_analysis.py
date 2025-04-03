import pandas as pd
import os
import json
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

# Ignore warning messages for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Define the path to the complete Yelp dataset
data_path = "yelp/"  # Modify this path to the location of your complete Yelp dataset

# ---------------------------
# 1. Preprocessing (Using the complete business dataset)
# ---------------------------

# Load the complete business dataset
business_df = pd.read_json(f"{data_path}/yelp_academic_dataset_business.json", lines=True)

# Load the check-in dataset to analyze check-in frequencies
checkin_df = pd.read_json(f"{data_path}/yelp_academic_dataset_checkin.json", lines=True)

# Extract the primary business category from the 'categories' column
# Categories are stored as a comma-separated string, we extract the first category as the main category
business_df['main_category'] = business_df['categories'].apply(
    lambda x: x.split(', ')[0] if isinstance(x, str) else None
)


# Extract 'price_range' from the 'attributes' column
def extract_price_range(attr):
    if isinstance(attr, dict):
        return attr.get('RestaurantsPriceRange2', None)
    if isinstance(attr, str):  # Handle JSON-like string attributes
        try:
            attr_dict = json.loads(attr.replace("'", "\""))  # Convert string to JSON-compatible format
            return attr_dict.get('RestaurantsPriceRange2', None)
        except:
            return None
    return None


# Apply the function to extract 'price_range' attribute from the business data
business_df['price_range'] = business_df['attributes'].apply(extract_price_range)

# Extracting check-in frequencies
# Counting the number of check-ins based on the 'date' column (comma-separated entries)
checkin_df['checkin_count'] = checkin_df['date'].str.split(',').apply(len)

# Aggregating check-in counts by business_id
checkin_features = checkin_df.groupby('business_id')['checkin_count'].sum().reset_index()

# Merge check-in data with business data
merged_df = pd.merge(business_df, checkin_features, on='business_id', how='left')

# Replace missing values with zero
merged_df.fillna(0, inplace=True)

# Create labels for high and low ratings
# High Rating (Label = 1): average stars >= 4
# Low Rating (Label = 0): average stars <= 2
merged_df['label'] = merged_df['stars'].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else np.nan))

# Remove rows where the label is NaN (ratings between 2 and 4 are ignored)
merged_df.dropna(subset=['label'], inplace=True)

# ---------------------------
# 2. Feature Value Probability Analysis (with minimum support filtering)
# ---------------------------

# Define the minimum support count to filter features with insufficient representation
MIN_SUPPORT_COUNT = 100  # Adjust this value based on dataset size


# Function to calculate value probabilities for a given feature
def calculate_value_probability(df, feature):
    # Count occurrences of each feature value with corresponding labels (0 or 1)
    value_counts = df.groupby([feature, 'label']).size().reset_index(name='count')

    # Count total occurrences of each feature value
    total_counts = df.groupby(feature).size().reset_index(name='total_count')

    # Merge the counts to compute probabilities
    result = pd.merge(value_counts, total_counts, on=feature)

    # Filter out entries with total count less than the defined minimum support count
    result = result[result['total_count'] >= MIN_SUPPORT_COUNT]

    # Calculate probability of each value-label pair
    result['probability'] = result['count'] / result['total_count']

    # Separate results for high ratings and low ratings
    high_prob = result[result['label'] == 1].sort_values(by='probability', ascending=False)
    low_prob = result[result['label'] == 0].sort_values(by='probability', ascending=False)

    return high_prob, low_prob


# Specify which features to analyze
features_to_check = ['main_category', 'price_range', 'city']

# Display results for each feature
for feature in features_to_check:
    high_probs, low_probs = calculate_value_probability(merged_df, feature)
    print(f"\nFeature: {feature}")
    print("High Rating Probabilities:\n", high_probs.head(10))
    print("Low Rating Probabilities:\n", low_probs.head(10))

# ---------------------------
# 3. Attribute Combinations Using Apriori Algorithm
# ---------------------------

# Convert 'price_range' to string for compatibility with attribute combination
merged_df['price_range'] = merged_df['price_range'].astype(str)

# Create a new column combining 'main_category' and 'price_range' for analysis
merged_df['attribute_combo'] = merged_df['main_category'].astype(str) + '_' + merged_df['price_range']

# One-hot encoding for each unique combination of attributes
one_hot = pd.get_dummies(merged_df['attribute_combo'])

# Perform Apriori algorithm to find frequent itemsets
# Minimum support threshold is set to 0.01 (1%)
frequent_itemsets = apriori(one_hot, min_support=0.01, use_colnames=True)

# Generate association rules based on frequent itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Sort the rules by lift score in descending order
rules = rules.sort_values(by='lift', ascending=False)

# Display the top 20 association rules
print("\nTop 20 Association Rules Related to High/Low Ratings:")
print(rules.head(20))
