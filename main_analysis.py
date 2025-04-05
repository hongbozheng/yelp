import pandas as pd
import os
import json
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
            attr_dict = json.loads(attr.replace("'", "\""))
            return attr_dict.get('RestaurantsPriceRange2', None)
        except:
            return None
    return None


business_df['price_range'] = business_df['attributes'].apply(extract_price_range)

# Extract check-in frequencies: count the number of check-ins for each business
checkin_df['checkin_count'] = checkin_df['date'].str.split(',').apply(len)
checkin_features = checkin_df.groupby('business_id')['checkin_count'].sum().reset_index()

# Merge check-in data with business data
merged_df = pd.merge(business_df, checkin_features, on='business_id', how='left')
merged_df.fillna(0, inplace=True)

# Create three-level labels:
# Low (0): stars <= 2, Medium (1): 2 < stars < 4, High (2): stars >= 4
merged_df['label'] = merged_df['stars'].apply(lambda x: 2 if x >= 4 else (0 if x <= 2 else 1))
# Print count of medium ratings to verify
print("Medium rating count:", merged_df[merged_df['label'] == 1].shape[0])
print("merged_df columns:", merged_df.columns)


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

# Bin continuous features into 4 quantile-based bins
merged_df['operation_hour_bins'] = pd.qcut(merged_df['average_open_hours'], q=4,
                                           labels=['Low', 'Medium', 'High', 'Very High'])
merged_df['checkin_count_range'] = pd.qcut(merged_df['checkin_count'], q=4,
                                           labels=['Low', 'Medium', 'High', 'Very High'])

# ---------------------------
# 3. Feature Value Probability Analysis
# ---------------------------

def compute_required_min_support(se_threshold=0.05):
    """
    Computes the minimum support count required so that the worst-case standard error
    (which occurs at p=0.5) is below se_threshold.

    Parameters:
    - se_threshold (float): The desired maximum standard error.

    Returns:
    - required_count (int): The minimum count required.
    """
    required_count = 0.25 / (se_threshold ** 2)
    return int(np.ceil(required_count))


# Example: set a desired standard error threshold (e.g., 0.05)
se_threshold = 0.03
min_support_required = compute_required_min_support(se_threshold)
print(
    f"To achieve a worst-case standard error below {se_threshold}, each feature value should have at least {min_support_required} observations.")


MIN_SUPPORT_COUNT = min_support_required  # Minimum support count threshold


def calculate_value_probability(df_filtered, df_total, feature):
    """
    Compute the conditional probability of each feature value leading to a particular label.
    df_filtered: DataFrame with only low and high ratings (labels 0 and 2, and medium 1 as well if present)
    df_total: Full DataFrame including all ratings for computing total counts.
    """
    value_counts = df_filtered.groupby([feature, 'label'], observed=False).size().reset_index(name='count')
    total_counts = df_total.groupby(feature, observed=False).size().reset_index(name='total_count')
    result = pd.merge(value_counts, total_counts, on=feature)
    result = result[result['total_count'] >= MIN_SUPPORT_COUNT]
    result['probability'] = result['count'] / result['total_count']
    # Get probabilities for each label separately
    prob_low = result[result['label'] == 0].sort_values(by='probability', ascending=False)
    prob_med = result[result['label'] == 1].sort_values(by='probability', ascending=False)
    prob_high = result[result['label'] == 2].sort_values(by='probability', ascending=False)
    return prob_low, prob_med, prob_high


# Use the full merged_df for total counts
merged_df_all = merged_df.copy()

features_to_check = ['main_category', 'price_range', 'city', 'operation_hour_bins', 'checkin_count_range']
for feature in features_to_check:
    low_probs, med_probs, high_probs = calculate_value_probability(merged_df, merged_df_all, feature)
    print(f"\nFeature: {feature}")
    print("Low Rating Probabilities:\n", low_probs.head(10))
    print("Medium Rating Probabilities:\n", med_probs.head(10))
    print("High Rating Probabilities:\n", high_probs.head(10))

    # For features with many unique values (main_category and city),
    # we plot two charts: one for the top 10 (highest high rating probability) and one for the bottom 10 (lowest high rating probability)
    if feature in ['main_category', 'city']:
        combined = pd.concat([low_probs, med_probs, high_probs])
        pivot_df = combined.pivot(index=feature, columns='label', values='probability').fillna(0)
        pivot_df = pivot_df.rename(columns={0: 'Low Rating', 1: 'Medium Rating', 2: 'High Rating'})
        # Sort by High Rating probability in ascending order
        pivot_sorted = pivot_df.sort_values(by='High Rating', ascending=True)

        top_10 = pivot_sorted.tail(10).sort_values(by='High Rating', ascending=True)
        bottom_10 = pivot_sorted.head(10).sort_values(by='High Rating', ascending=True)

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12), sharex=True)
        top_10.plot(kind='barh', ax=axes[0],
                    color={'High Rating': 'green', 'Medium Rating': 'blue', 'Low Rating': 'red'})
        axes[0].set_title(f"Top 10 {feature} Values (Highest High Rating Probability)", fontsize=18)
        axes[0].set_xlabel("Probability", fontsize=16)
        axes[0].set_ylabel(feature, fontsize=16)
        axes[0].tick_params(axis='x', labelsize=14)
        axes[0].tick_params(axis='y', labelsize=14)
        axes[0].legend(fontsize=14, loc='lower right')

        bottom_10.plot(kind='barh', ax=axes[1],
                       color={'High Rating': 'green', 'Medium Rating': 'blue', 'Low Rating': 'red'})
        axes[1].set_title(f"Bottom 10 {feature} Values (Lowest High Rating Probability)", fontsize=18)
        axes[1].set_xlabel("Probability", fontsize=16)
        axes[1].set_ylabel(feature, fontsize=16)
        axes[1].tick_params(axis='x', labelsize=14)
        axes[1].tick_params(axis='y', labelsize=14)
        axes[1].legend(fontsize=14, loc='lower right')

        plt.figtext(0.2, 0.5, '... (omitted middle values) ...', ha='center', fontsize=12, color='black')
        plt.tight_layout()
        plt.show()
    else:
        # For features with fixed order (price_range, operation_hour_bins, checkin_count_range)
        if feature == 'price_range':
            order = ['1', '2', '3', '4']  # Adjust as necessary
        else:
            order = ['Low', 'Medium', 'High', 'Very High']
        combined = pd.concat([low_probs, med_probs, high_probs])
        pivot_df = combined.pivot(index=feature, columns='label', values='probability').fillna(0)
        pivot_df = pivot_df.rename(columns={0: 'Low Rating', 1: 'Medium Rating', 2: 'High Rating'})
        pivot_ordered = pivot_df.reindex(order)
        plt.figure(figsize=(10, 6))
        pivot_ordered.plot(kind='bar', rot=0)
        plt.title(f'Probability Analysis for {feature} (Ordered)')
        plt.xlabel(feature)
        plt.ylabel('Probability')
        plt.legend(title='Rating')
        plt.tight_layout()
        plt.show()


# ---------------------------
# 4. Information Gain Ratio Analysis
# ---------------------------
def calculate_information_gain_ratio(df, features, label_column='label'):
    label = df[label_column].values
    results = {}
    le = LabelEncoder()
    for feature in features:
        if df[feature].dtype == 'object' or 'category' in str(df[feature].dtype):
            feature_values = le.fit_transform(df[feature].astype(str))
        else:
            feature_values = df[feature].values
        feature_values = feature_values.reshape(-1, 1)
        info_gain = mutual_info_classif(feature_values, label, discrete_features=True)[0]
        if df[feature].dtype == 'object' or 'category' in str(df[feature].dtype):
            int_values = pd.Series(feature_values.flatten()).astype(int)
        else:
            int_values = pd.Series(feature_values.flatten())
        value_counts = np.bincount(int_values)
        probs = value_counts / len(int_values)
        entropy = -np.sum(probs * np.log2(probs + 1e-9))
        if entropy != 0:
            info_gain_ratio = info_gain / entropy
        else:
            info_gain_ratio = 0
        results[feature] = info_gain_ratio
    return results


info_gain_scores = calculate_information_gain_ratio(merged_df, features_to_check)
info_gain_sorted = sorted(info_gain_scores.items(), key=lambda x: x[1], reverse=True)
info_gain_df = pd.DataFrame(info_gain_sorted, columns=['Feature', 'Info_Gain_Ratio'])

plt.figure(figsize=(12, 6))
sns.barplot(x='Feature', y='Info_Gain_Ratio', data=info_gain_df, palette='viridis')
plt.title('Information Gain Ratio for Selected Features', fontsize=18)
plt.xlabel('Feature', fontsize=18)
plt.ylabel('Information Gain Ratio', fontsize=18)
plt.xticks(rotation=45, fontsize=16)  # x-tick font size and rotation
plt.yticks(fontsize=16)               # y-tick font size
plt.tight_layout()
plt.show()

# ---------------------------
# 5. Attribute Combinations Using Apriori Algorithm
# ---------------------------
# merged_df['price_range'] = merged_df['price_range'].astype(str)
# merged_df['attribute_combo'] = (merged_df['main_category'].astype(str) + '_' +
#                                 merged_df['price_range'] + '_' +
#                                 merged_df['checkin_count_range'].astype(str) + '_' +
#                                 merged_df['operation_hour_bins'].astype(str) + '_' +
#                                 merged_df['city'].astype(str))
#
# one_hot = pd.get_dummies(merged_df['attribute_combo'])
# from scipy.sparse import csr_matrix
#
# one_hot_sparse = pd.DataFrame.sparse.from_spmatrix(csr_matrix(one_hot.values), columns=one_hot.columns)
# one_hot_sparse = one_hot_sparse.astype(bool)
#
# min_support_value = 0.001  # Lower threshold to capture more itemsets
# frequent_itemsets = apriori(one_hot_sparse, min_support=min_support_value, use_colnames=True, low_memory=True)
# if not frequent_itemsets.empty:
#     frequent_itemsets['num_features'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
#     filtered_itemsets = frequent_itemsets[
#         (frequent_itemsets['num_features'] >= 5) & (frequent_itemsets['num_features'] <= 6)]
#     if not filtered_itemsets.empty:
#         try:
#             rules = association_rules(filtered_itemsets, metric="lift", min_threshold=1.0, support_only=True)
#             rules = rules.sort_values(by='lift', ascending=False)
#             print("\nTop 20 Association Rules Related to High/Low Ratings:")
#             print(rules.head(20))
#         except KeyError as e:
#             print(f"KeyError during rule generation: {e}")
#     else:
#         print("No frequent itemsets of size 5 or 6 found. Consider lowering the min_support threshold.")
# else:
#     print("No frequent itemsets found. Consider lowering the min_support threshold.")
