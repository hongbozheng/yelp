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

business_df['price_range'] = business_df['attributes'].apply(extract_price_range)

# Extracting check-in frequencies
checkin_df['checkin_count'] = checkin_df['date'].str.split(',').apply(len)
checkin_features = checkin_df.groupby('business_id')['checkin_count'].sum().reset_index()

# Merge check-in data with business data
merged_df = pd.merge(business_df, checkin_features, on='business_id', how='left')
merged_df.fillna(0, inplace=True)

# Create labels for high and low ratings
merged_df['label'] = merged_df['stars'].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else np.nan))
merged_df.dropna(subset=['label'], inplace=True)

# ---------------------------
# 2. Operation Hour Analysis
# ---------------------------

def extract_hour_length(hours):
    if isinstance(hours, dict):
        total_hours = 0
        for day, time_range in hours.items():
            try:
                open_time, close_time = time_range.split('-')
                open_hour, open_min = map(int, open_time.split(':'))
                close_hour, close_min = map(int, close_time.split(':'))
                open_minutes = open_hour * 60 + open_min
                close_minutes = close_hour * 60 + close_min
                if close_minutes < open_minutes:  # Handle overnight hours
                    close_minutes += 24 * 60
                total_hours += (close_minutes - open_minutes) / 60
            except:
                continue
        return total_hours / 7  # Average operation hours per day
    return None

# Extract operation hour information
business_df['average_open_hours'] = business_df['hours'].apply(extract_hour_length)
merged_df = pd.merge(merged_df, business_df[['business_id', 'average_open_hours']], on='business_id', how='left')

# Binning operation hours into 4 categories
merged_df['operation_hour_bins'] = pd.qcut(merged_df['average_open_hours'].fillna(0), q=4,
                                           labels=['Low', 'Medium', 'High', 'Very High'])

# ---------------------------
# 3. Check-in Count Analysis (Binning)
# ---------------------------

# Binning check-in counts into 4 categories
merged_df['checkin_count_range'] = pd.qcut(merged_df['checkin_count'].fillna(0), q=4,
                                           labels=['Low', 'Medium', 'High', 'Very High'])

# ---------------------------
# 4. Feature Value Probability Analysis (with minimum support filtering)
# ---------------------------

MIN_SUPPORT_COUNT = 200  # Set a minimum support count threshold

def calculate_value_probability(df, feature):
    # Count occurrences of each feature value with corresponding labels (0 or 1)
    value_counts = df.groupby([feature, 'label'], observed=False).size().reset_index(name='count')

    # Count total occurrences of each feature value
    total_counts = df.groupby(feature, observed=False).size().reset_index(name='total_count')

    # Merge the counts to compute probabilities
    result = pd.merge(value_counts, total_counts, on=feature)
    result = result[result['total_count'] >= MIN_SUPPORT_COUNT]
    result['probability'] = result['count'] / result['total_count']
    high_prob = result[result['label'] == 1].sort_values(by='probability', ascending=False)
    low_prob = result[result['label'] == 0].sort_values(by='probability', ascending=False)
    return high_prob, low_prob

features_to_check = ['main_category', 'price_range', 'city', 'operation_hour_bins', 'checkin_count_range']
for feature in features_to_check:
    high_probs, low_probs = calculate_value_probability(merged_df, feature)
    print(f"\nFeature: {feature}")
    print("High Rating Probabilities:\n", high_probs.head(10))
    print("Low Rating Probabilities:\n", low_probs.head(10))

# # ---------------------------
# # 5. Attribute Combinations Using Apriori Algorithm
# # ---------------------------
#
# merged_df['price_range'] = merged_df['price_range'].astype(str)
#
# # Create a combined attribute column for Apriori analysis
# merged_df['attribute_combo'] = merged_df['main_category'].astype(str) + '_' + merged_df['price_range'] + '_' + \
#                                merged_df['checkin_count_range'].astype(str) + '_' + merged_df['operation_hour_bins'].astype(str)
#
# # One-hot encoding for each unique combination of attributes
# one_hot = pd.get_dummies(merged_df['attribute_combo'])
#
# # Perform Apriori algorithm to find frequent itemsets
# frequent_itemsets = apriori(one_hot, min_support=0.01, use_colnames=True)
#
# # Generate association rules based on frequent itemsets
# rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
# rules = rules.sort_values(by='lift', ascending=False)
#
# # Display the top 20 association rules
# print("\nTop 20 Association Rules Related to High/Low Ratings:")
# print(rules.head(20))
