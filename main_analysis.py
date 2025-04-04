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

# Extracting check-in frequencies: count the number of check-ins for each business
checkin_df['checkin_count'] = checkin_df['date'].str.split(',').apply(len)
checkin_features = checkin_df.groupby('business_id')['checkin_count'].sum().reset_index()

# Merge check-in data with business data
merged_df = pd.merge(business_df, checkin_features, on='business_id', how='left')
merged_df.fillna(0, inplace=True)

# Create labels for high and low ratings:
# High Rating (Label = 1): stars >= 4; Low Rating (Label = 0): stars <= 2;
merged_df['label'] = merged_df['stars'].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else np.nan))
merged_df.dropna(subset=['label'], inplace=True)
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
MIN_SUPPORT_COUNT = 200  # Minimum support count threshold


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

    # Combine high and low probability results
    combined = pd.concat([high_probs, low_probs])
    pivot_df = combined.pivot(index=feature, columns='label', values='probability').fillna(0)
    pivot_df = pivot_df.rename(columns={0: 'Low Rating', 1: 'High Rating'})

    if feature in ['main_category', 'city']:
        # For features with many values, create a pivot table for the high rating probability
        combined = pd.concat([high_probs, low_probs])
        pivot_df = combined.pivot(index=feature, columns='label', values='probability').fillna(0)
        pivot_df = pivot_df.rename(columns={0: 'Low Rating', 1: 'High Rating'})
        # Sort descending by High Rating probability
        pivot_sorted = pivot_df.sort_values(by='High Rating', ascending=False)

        # Top 10: values with the highest High Rating probabilities
        top_10 = pivot_sorted.head(10).sort_values(by='High Rating', ascending=True)
        # Bottom 10: values with the lowest High Rating probabilities
        bottom_10 = pivot_sorted.tail(10).sort_values(by='High Rating', ascending=True)

        # Create vertical subplots: top 10 on top and bottom 10 on bottom
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

        # Plot top 10 (highest high rating probability) on the first subplot
        top_10.plot(kind='barh', ax=axes[0], legend=True, color={'High Rating': 'green', 'Low Rating': 'red'})
        axes[0].set_title(f"Top 10 {feature} Values (Highest High Rating Probability)")
        axes[0].set_xlabel("Probability")
        axes[0].set_ylabel(feature)

        # Plot bottom 10 (lowest high rating probability) on the second subplot
        bottom_10.plot(kind='barh', ax=axes[1], legend=True, color={'High Rating': 'green', 'Low Rating': 'red'})
        axes[1].set_title(f"Bottom 10 {feature} Values (Lowest High Rating Probability)")
        axes[1].set_xlabel("Probability")
        axes[1].set_ylabel(feature)

        # Add annotation to indicate omitted middle values
        plt.figtext(0.5, 0.48, '... (omitted middle values) ...', ha='center', fontsize=12, color='gray')
        plt.tight_layout()
        plt.show()
    else:
        # For features with fixed order (price_range, operation_hour_bins, checkin_count_range),
        # we reindex according to the desired order.
        if feature == 'price_range':
            order = ['1', '2', '3', '4']  # Adjust as needed
        else:
            order = ['Low', 'Medium', 'High', 'Very High']
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

plt.figure(figsize=(10, 6))
sns.barplot(x='Feature', y='Info_Gain_Ratio', data=info_gain_df, palette='viridis')
plt.title('Information Gain Ratio for Selected Features')
plt.xlabel('Feature')
plt.ylabel('Information Gain Ratio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------
# 5. Attribute Combinations Using Apriori Algorithm
# ---------------------------
merged_df['price_range'] = merged_df['price_range'].astype(str)
merged_df['attribute_combo'] = (merged_df['main_category'].astype(str) + '_' +
                                merged_df['price_range'] + '_' +
                                merged_df['checkin_count_range'].astype(str) + '_' +
                                merged_df['operation_hour_bins'].astype(str) + '_' +
                                merged_df['city'].astype(str))

one_hot = pd.get_dummies(merged_df['attribute_combo'])
from scipy.sparse import csr_matrix

one_hot_sparse = pd.DataFrame.sparse.from_spmatrix(csr_matrix(one_hot.values), columns=one_hot.columns)
one_hot_sparse = one_hot_sparse.astype(bool)

min_support_value = 0.001  # Lower threshold to capture more itemsets
frequent_itemsets = apriori(one_hot_sparse, min_support=min_support_value, use_colnames=True, low_memory=True)
if not frequent_itemsets.empty:
    frequent_itemsets['num_features'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    filtered_itemsets = frequent_itemsets[
        (frequent_itemsets['num_features'] >= 5) & (frequent_itemsets['num_features'] <= 6)]
    if not filtered_itemsets.empty:
        try:
            rules = association_rules(filtered_itemsets, metric="lift", min_threshold=1.0, support_only=True)
            rules = rules.sort_values(by='lift', ascending=False)
            print("\nTop 20 Association Rules Related to High/Low Ratings:")
            print(rules.head(20))
        except KeyError as e:
            print(f"KeyError during rule generation: {e}")
    else:
        print("No frequent itemsets of size 5 or 6 found. Consider lowering the min_support threshold.")
else:
    print("No frequent itemsets found. Consider lowering the min_support threshold.")
