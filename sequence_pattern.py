#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sequence_pattern_analysis.py

分析高评分（High）和低评分（Low）商家的属性序列模式。
使用 PrefixSpan 算法挖掘 "main_category -> price_range -> checkin_count_range -> operation_hour_bins" 上的频繁序列，
并对比在高评分和低评分业务中的支持度。

依赖：
    pip install prefixspan
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import List
from prefixspan import PrefixSpan

def parse_attributes(attr):
    """
    解析 attributes 字段，将其转换为统一的 flat dict，包括嵌套结构如 BusinessParking
    """
    result = {}
    if isinstance(attr, str):
        try:
            attr = json.loads(attr.replace("'", '"'))
        except:
            return result
    if not isinstance(attr, dict):
        return result

    for key, val in attr.items():
        if val in ['True', 'False']:
            val = val == 'True'
        elif isinstance(val, str) and val.startswith("{") and key == "BusinessParking":
            try:
                val = json.loads(val.replace("'", '"'))
                for sub_key, sub_val in val.items():
                    result[f"{key}_{sub_key}"] = sub_val
                continue
            except:
                continue
        result[key] = val
    return result

def load_and_preprocess(data_path: str) -> pd.DataFrame:
    """
    加载 Yelp Business 和 Checkin 数据，合并并生成所需特征和标签。
    返回包含：
      - main_category
      - price_range
      - checkin_count
      - average_open_hours
      - operation_hour_bins
      - checkin_count_range
      - label (0=Low,1=Medium,2=High)
    """
    business_df = pd.read_json(f"{data_path}/yelp_academic_dataset_business.json", lines=True)
    checkin_df = pd.read_json(f"{data_path}/yelp_academic_dataset_checkin.json", lines=True)

    business_df['main_category'] = business_df['categories'].apply(
        lambda x: x.split(', ')[0] if isinstance(x, str) else None
    )

    # 展开 attributes 字段
    attr_df = business_df['attributes'].apply(parse_attributes).apply(pd.Series)
    business_df = pd.concat([business_df.drop(columns=['attributes']), attr_df], axis=1)

    # 修复 WiFi 字段的字符串格式
    if 'WiFi' in business_df.columns:
        business_df['WiFi'] = business_df['WiFi'].astype(str).str.extract(r"(no|free|paid)", expand=False)

    # 提取价格区间字段（已在展开中）
    if 'RestaurantsPriceRange2' in business_df.columns:
        business_df.rename(columns={'RestaurantsPriceRange2': 'price_range'}, inplace=True)

    # 统计 check-in 次数
    checkin_df['checkin_count'] = checkin_df['date'].str.split(',').apply(len)
    chk = checkin_df.groupby('business_id')['checkin_count'].sum().reset_index()
    merged = pd.merge(business_df, chk, on='business_id', how='left').fillna({'checkin_count': 0})

    # 打标签
    merged['label'] = merged['stars'].apply(lambda x: 2 if x >= 4 else (0 if x <= 2 else 1))

    # 营业时间
    def avg_open_hours(hours):
        if not isinstance(hours, dict): return np.nan
        total = 0
        for t in hours.values():
            try:
                o, c = t.split('-')
                oh, om = map(int, o.split(':'))
                ch, cm = map(int, c.split(':'))
                omins, cmins = oh * 60 + om, ch * 60 + cm
                if cmins < omins: cmins += 24 * 60
                total += (cmins - omins) / 60
            except: continue
        return total / 7

    merged['average_open_hours'] = merged['hours'].apply(avg_open_hours)

    merged['operation_hour_bins'] = pd.qcut(
        merged['average_open_hours'].fillna(0),
        q=4,
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    merged['checkin_count_range'] = pd.qcut(
        merged['checkin_count'],
        q=4,
        labels=['Low', 'Medium', 'High', 'Very High']
    )

    # 输出示例记录供检查
    print("\n📌 Sample merged business record:")
    print(merged.head(1).T)

    return merged

def generate_sequences(df: pd.DataFrame, feature_cols: List[str]) -> (List[List[str]], List[List[str]]):
    df = df.dropna(subset=feature_cols + ['label'])
    seqs_high = df[df['label'] == 2][feature_cols].astype(str).values.tolist()
    seqs_low  = df[df['label'] == 0][feature_cols].astype(str).values.tolist()
    return seqs_high, seqs_low

def mine_sequences(sequences: List[List[str]], min_support: int, min_len: int = 1, max_len: int = 10) -> List[tuple]:
    ps = PrefixSpan(sequences)
    patterns = ps.frequent(min_support)
    filtered = [p for p in patterns if min_len <= len(p[1]) <= max_len]
    return sorted(filtered, key=lambda x: x[0], reverse=True)

def visualize_sequence_support(patterns_high: List[tuple], patterns_low: List[tuple], top_n: int = 10):
    top_high = patterns_high[:top_n]
    sup_high = [s for s, _ in top_high]
    pats = [tuple(p) for _, p in top_high]
    low_dict = {tuple(p): s for s, p in patterns_low}
    sup_low = [low_dict.get(p, 0) for p in pats]
    labels = [' -> '.join(p) for p in pats]

    if not labels:
        print("⚠️ No frequent sequences found with the specified length and support threshold.")
        return

    df_plot = pd.DataFrame({
        'High Support': sup_high,
        'Low Support' : sup_low
    }, index=labels)

    df_plot.plot(kind='bar', figsize=(12, 6))
    plt.title('Top Sequence Support in High vs Low Ratings')
    plt.ylabel('Support Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    data_path = 'data'
    merged = load_and_preprocess(data_path)
    features = ['main_category', 'price_range', 'checkin_count_range', 'operation_hour_bins']

    seqs_high, seqs_low = generate_sequences(merged, features)
    min_support = 5

    patterns_high = mine_sequences(seqs_high, min_support, min_len=5, max_len=10)
    patterns_low  = mine_sequences(seqs_low,  min_support, min_len=3, max_len=10)

    print("\nTop 10 Frequent Sequences in High-Rated Businesses:")
    for sup, pat in patterns_high[:10]:
        print(f"  {pat} (support={sup})")

    print("\nTop 10 Frequent Sequences in Low-Rated Businesses:")
    for sup, pat in patterns_low[:10]:
        print(f"  {pat} (support={sup})")

    visualize_sequence_support(patterns_high, patterns_low, top_n=10)

if __name__ == '__main__':
    main()
