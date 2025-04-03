import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

# 忽略警告信息
warnings.filterwarnings("ignore", category=UserWarning)

# 定义数据路径
data_path = "yelp_illinois/"  # 修改为你的数据路径

# ---------------------------
# 1. 加载数据
# ---------------------------
review_df = pd.read_json(f"{data_path}/review_illinois.json", lines=True)

# ---------------------------
# 2. N-gram Analysis
# ---------------------------
def extract_phrases(df, column, ngram_range=(1, 1), top_n=30):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range, max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform(df[column].dropna())
    phrases = vectorizer.get_feature_names_out()
    return phrases

# 将评分分为高评分和低评分
high_rating_reviews = review_df[review_df['stars'] >= 4]
low_rating_reviews = review_df[review_df['stars'] <= 2]

high_phrases = {}
low_phrases = {}

ngram_ranges = {
    "1-gram": (1, 1),
    "2-gram": (2, 2),
    "3-gram": (3, 3),
    ">=5-gram": (5, 6)
}

# 提取不同 N-gram 的词组
for key, ngram_range in ngram_ranges.items():
    high_phrases[key] = extract_phrases(high_rating_reviews, 'text', ngram_range=ngram_range, top_n=30)
    low_phrases[key] = extract_phrases(low_rating_reviews, 'text', ngram_range=ngram_range, top_n=30)

# 输出结果
print("\nN-gram Analysis Results:")
for key in ngram_ranges.keys():
    print(f"\nHigh Rating Phrases ({key}):\n{high_phrases[key]}")
    print(f"\nLow Rating Phrases ({key}):\n{low_phrases[key]}")
