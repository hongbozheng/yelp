import pandas as pd
import os
import json
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

# 忽略警告信息
warnings.filterwarnings("ignore", category=UserWarning)

# 定义数据路径 (现在使用完整的数据集)
data_path = "yelp/"  # 修改为你的完整数据集路径

# ---------------------------
# 1. Preprocessing (完整商家数据)
# ---------------------------
# 加载完整的商家数据集
business_df = pd.read_json(f"{data_path}/yelp_academic_dataset_business.json", lines=True)
checkin_df = pd.read_json(f"{data_path}/yelp_academic_dataset_checkin.json", lines=True)

# 提取商家的主要类别
business_df['main_category'] = business_df['categories'].apply(
    lambda x: x.split(', ')[0] if isinstance(x, str) else None
)

# 提取 price_range 从 attributes 列
def extract_price_range(attr):
    if isinstance(attr, dict):
        return attr.get('RestaurantsPriceRange2', None)
    if isinstance(attr, str):  # 处理 JSON 字符串格式的属性
        try:
            attr_dict = json.loads(attr.replace("'", "\""))
            return attr_dict.get('RestaurantsPriceRange2', None)
        except:
            return None
    return None

business_df['price_range'] = business_df['attributes'].apply(extract_price_range)

# 提取 Check-in 频率
checkin_df['checkin_count'] = checkin_df['date'].str.split(',').apply(len)
checkin_features = checkin_df.groupby('business_id')['checkin_count'].sum().reset_index()

# 合并 Check-in 数据
merged_df = pd.merge(business_df, checkin_features, on='business_id', how='left')
merged_df.fillna(0, inplace=True)

# 创建标签
merged_df['label'] = merged_df['stars'].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else np.nan))
merged_df.dropna(subset=['label'], inplace=True)

# ---------------------------
# 2. Feature Value Probability Analysis (增加最小样本数过滤)
# ---------------------------

# 定义最小支持样本数
MIN_SUPPORT_COUNT = 100  # 提高支持样本数以保证更高的代表性

def calculate_value_probability(df, feature):
    value_counts = df.groupby([feature, 'label']).size().reset_index(name='count')
    total_counts = df.groupby(feature).size().reset_index(name='total_count')
    result = pd.merge(value_counts, total_counts, on=feature)

    # 只保留 total_count >= MIN_SUPPORT_COUNT 的项
    result = result[result['total_count'] >= MIN_SUPPORT_COUNT]

    # 计算概率
    result['probability'] = result['count'] / result['total_count']

    high_prob = result[result['label'] == 1].sort_values(by='probability', ascending=False)
    low_prob = result[result['label'] == 0].sort_values(by='probability', ascending=False)
    return high_prob, low_prob

features_to_check = ['main_category', 'price_range', 'city']
for feature in features_to_check:
    high_probs, low_probs = calculate_value_probability(merged_df, feature)
    print(f"\nFeature: {feature}")
    print("High Rating Probabilities:\n", high_probs.head(10))
    print("Low Rating Probabilities:\n", low_probs.head(10))

# ---------------------------
# 3. Attribute Combinations Using Apriori Algorithm
# ---------------------------
merged_df['price_range'] = merged_df['price_range'].astype(str)
merged_df['attribute_combo'] = merged_df['main_category'].astype(str) + '_' + merged_df['price_range']

# 构建频繁项集
one_hot = pd.get_dummies(merged_df['attribute_combo'])
frequent_itemsets = apriori(one_hot, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values(by='lift', ascending=False)

# 显示最重要的规则
print("\nTop 20 Association Rules Related to High/Low Ratings:")
print(rules.head(20))
