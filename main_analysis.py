import pandas as pd
import os
import json
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
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
# 2. Operation Hours Analysis (Discretization)
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
                if close_minutes < open_minutes:  # Handle cases where business hours cross midnight
                    close_minutes += 24 * 60
                total_hours += (close_minutes - open_minutes) / 60
            except:
                continue
        return total_hours / 7  # Average hours per day
    return None


business_df['average_open_hours'] = business_df['hours'].apply(extract_hour_length)
merged_df = pd.merge(merged_df, business_df[['business_id', 'average_open_hours']], on='business_id', how='left')
merged_df.fillna(0, inplace=True)

# Bin continuous features (operation hours and checkin counts)
merged_df['operation_hour_bins'] = pd.qcut(merged_df['average_open_hours'], q=4,
                                           labels=['Low', 'Medium', 'High', 'Very High'])
merged_df['checkin_count_range'] = pd.qcut(merged_df['checkin_count'], q=4,
                                           labels=['Low', 'Medium', 'High', 'Very High'])

# ---------------------------
# 3. Feature Value Probability Analysis
# ---------------------------

MIN_SUPPORT_COUNT = 200


def calculate_value_probability(df, feature):
    value_counts = df.groupby([feature, 'label'], observed=False).size().reset_index(name='count')
    total_counts = df.groupby(feature, observed=False).size().reset_index(name='total_count')
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


# ---------------------------
# 4. Information Gain Ratio Analysis
# ---------------------------

def calculate_information_gain_ratio(df, features, label_column='label'):
    label = df[label_column].values
    results = {}
    label_encoder = LabelEncoder()

    for feature in features:
        if df[feature].dtype == 'object' or 'category' in str(df[feature].dtype):
            feature_values = label_encoder.fit_transform(df[feature].astype(str))
        else:
            feature_values = df[feature].values

        info_gain = mutual_info_classif(feature_values.reshape(-1, 1), label, discrete_features=True)[0]

        value_counts = np.bincount(feature_values)
        probs = value_counts / len(feature_values)
        entropy = -np.sum(probs * np.log2(probs + 1e-9))

        if entropy != 0:
            info_gain_ratio = info_gain / entropy
        else:
            info_gain_ratio = 0

        results[feature] = info_gain_ratio

    return results


features_to_check = ['main_category', 'price_range', 'city', 'operation_hour_bins', 'checkin_count_range']
info_gain_scores = calculate_information_gain_ratio(merged_df, features_to_check)
info_gain_sorted = sorted(info_gain_scores.items(), key=lambda x: x[1], reverse=True)

print("\nInformation Gain Ratio Scores (Sorted):")
for feature, score in info_gain_sorted:
    print(f"{feature}: {score:.4f}")

# ---------------------------
# 5. Attribute Combinations Using Apriori Algorithm
# ---------------------------

# Create attribute combinations using relevant features
merged_df['attribute_combo'] = merged_df['main_category'].astype(str) + '_' + \
                               merged_df['price_range'].astype(str) + '_' + \
                               merged_df['checkin_count_range'].astype(str) + '_' + \
                               merged_df['operation_hour_bins'].astype(str) + '_' + \
                               merged_df['city'].astype(str)

# One-hot encoding of the attribute combinations
one_hot = pd.get_dummies(merged_df['attribute_combo'])

# Check if one_hot DataFrame is non-empty
if not one_hot.empty:
    # Adjust min_support to ensure frequent itemsets are generated
    min_support_value = 0.0005  # Reduced to increase the likelihood of generating rules
    frequent_itemsets = apriori(one_hot, min_support=min_support_value, use_colnames=True)

    if not frequent_itemsets.empty:
        # Filter frequent itemsets to keep only those with 5 or 6 features
        frequent_itemsets['num_features'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        filtered_itemsets = frequent_itemsets[
            (frequent_itemsets['num_features'] >= 5) & (frequent_itemsets['num_features'] <= 6)]

        if not filtered_itemsets.empty:
            # Generate association rules from the filtered frequent itemsets
            rules = association_rules(filtered_itemsets, metric="lift", min_threshold=1.0)
            rules = rules.sort_values(by='lift', ascending=False)

            # Display top 20 rules if available
            if len(rules) > 0:
                print("\nTop 20 Association Rules Related to High/Low Ratings:")
                print(rules.head(20))
            else:
                print("No association rules were generated with the specified parameters.")
        else:
            print("No frequent itemsets of size 5 or 6 found. Try reducing the min_support threshold.")
    else:
        print("No frequent itemsets found. Try reducing the min_support threshold.")
else:
    print("One-hot encoded DataFrame is empty. Check your data processing steps.")