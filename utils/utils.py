from pandas import DataFrame
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns


def parse_elite(elite):
    if pd.isna(elite) or elite.strip() == '':
        return 0
    yrs = elite.replace("20,20", "2020").split(sep=',')
    yrs = {yr.strip() for yr in yrs if yr.strip().isdigit() and len(yr.strip()) == 4}
    return len(yrs)


def review_feature(
        review_fp: str,
        business_fp: str,
        user_fp: str,
        checkin_fp: str,
        tip_fp: str,
        top_k: int,
        min_useful: int,
):
    print("📂 [INFO] Loading reviews...")
    df = pd.read_json(path_or_buf=review_fp, lines=True)
    print("🧹 [INFO] Converting features...")
    df['review'] = df['text'].apply(lambda x: len(x.split()))
    df['date'] = pd.to_datetime(df['date'], unit='ms')

    print("📂 [INFO] Loading businesses...")
    business_df = pd.read_json(path_or_buf=business_fp, lines=True)
    business_df = business_df[['business_id', 'city', 'categories']].dropna()
    print("🔗 [INFO] Merging business to review...")
    df = df.merge(business_df, on='business_id', how='inner')
    print("🧹 [INFO] Converting categories...")
    df['categories'] = df['categories'].str.split(', ')

    print(f"🧹 [INFO] Using top-{top_k} categories...")
    top_cats = df['categories'].explode().dropna().value_counts().head(top_k) \
        .index.tolist()
    print(f"📊 [INFO] Binarize category features...")
    for cat in top_cats:
        df[cat] = df['categories'].apply(lambda x: int(cat in x))

    print("📂 [INFO] Loading user...")
    user_df = pd.read_json(path_or_buf=user_fp, lines=True)
    print("🔗 [INFO] Merging user to review...")
    df = df.merge(user_df, on='user_id', how='inner', suffixes=('', '_user'))
    print("🧹 [INFO] Converting features...")
    df['yelping_since'] = pd.to_datetime(df['yelping_since'], errors='coerce')
    df['elite'] = df['elite'].fillna('').apply(parse_elite)
    df['friend'] = df['friends'].fillna('').apply(
        lambda s: len(s.split(', ')) if s else 0
    )
    df['rev-age'] = (
            (df['date'] - df['yelping_since']) / pd.Timedelta(days=365)
    ).fillna(0).round(2)

    print("🧹 [INFO] Converting temporal features...")
    df['days_since_1st_rev'] = (
            df['date'] - df.groupby('user_id')['date'].transform('min')
    ).dt.days
    df['days_since_dataset_start'] = (
            df['date'] - pd.to_datetime("2019-01-01", errors='coerce')
    ).dt.days
    df['year_month'] = df['date'].dt.to_period('M')
    rev_freq_month = df.groupby(['user_id', 'year_month']).size() \
        .reset_index(name='rev_freq_month')
    df = df.merge(rev_freq_month, on=['user_id', 'year_month'], how='left')
    monthly_std = rev_freq_month.groupby('user_id')['rev_freq_month'].std() \
        .rename('rev_freq_std')
    df = df.merge(monthly_std, on='user_id', how='left')

    print("📂 [INFO] Loading check-in...")
    checkin_df = pd.read_json(path_or_buf=checkin_fp, lines=True)
    checkin_df['checkin_count'] = len(checkin_df['date'])
    checkin_sum = checkin_df.groupby('business_id')['checkin_count'].sum()
    print("🔗 [INFO] Adding check-in to review...")
    df['check-in'] = df['business_id'].map(checkin_sum).fillna(0)

    print("📂 [INFO] Loading tip...")
    tip_df = pd.read_json(tip_fp, lines=True)
    tip_counts = tip_df.groupby(['user_id', 'business_id']).size()
    print("🔗 [INFO] Adding tip to review...")
    df['tip_count'] = list(
        df.set_index(['user_id', 'business_id']).index.map(tip_counts).fillna(0)
    )

    df = df.dropna()

    print("🎯 [INFO] Creating target variable...")
    df['label'] = (df['useful'] >= min_useful).astype(int)
    total = df['label'].shape[0]
    pos = (df['label'] == 1).sum()
    neg = (df['label'] == 0).sum()
    print(f"📊 [INFO] Label Distribution:")
    print(f"[INFO] ➤ Helpful     [1]: {pos} {pos / total:.4%}")
    print(f"[INFO] ➤ Not helpful [0]: {neg} {neg / total:.4%}")

    print(f"✅ [INFO] Final review feature shape {df.shape}")

    return df, top_cats


def rule_feature(
        df: DataFrame,
        exclude_cols: List[str],
        percentile: float,
):
    print(f"🔁 [INFO] Performing top {int((1 - percentile) * 100)}% binarization...")
    exclude_cols.extend(['label', 'useful'])
    cols = df.select_dtypes(include='number').columns.difference(exclude_cols)

    print("📊 [INFO] Distribution of 0s and 1s in binarized columns:")
    for col in cols:
        if col in exclude_cols:
            continue
        thres = df[col].quantile(percentile)
        df[col] = (df[col] >= thres).astype(int)
        total = df[col].shape[0]
        ones = (df[col] == 1).sum()
        zeros = (df[col] == 0).sum()
        one_pct = ones / total
        zero_pct = zeros / total
        print(f"{col:<25}: 0s = {zeros:<6} [{zero_pct:8.4%}] | 1s = {ones:<6} [{one_pct:8.4%}]")

    df = df.dropna()

    print(f"✅ [INFO] Binarized {len(cols)} columns.")
    print(f"✅ [INFO] Final rule feature shape {df.shape}")

    return df


def user_feature(
        review_fp: str,
        business_fp: str,
        user_fp: str,
        checkin_fp: str,
        tip_fp: str,
        top_k: int,
        min_useful: int,
        useful_thres: float,
):
    df, top_cats = review_feature(
        review_fp=review_fp,
        business_fp=business_fp,
        user_fp=user_fp,
        checkin_fp=checkin_fp,
        tip_fp=tip_fp,
        top_k=top_k,
        min_useful=min_useful,
    )

    top_cats.extend([
        'review_id', 'business_id', 'text', 'date', 'city', 'categories',
        'name', 'yelping_since', 'friends', 'year_month', 'label',
    ])
    df = df.drop(columns=top_cats, errors='ignore')

    print("📊 [INFO] Aggregating user-level features via mean...")
    df = df.groupby('user_id').mean().reset_index()
    print(f"✅ [INFO] Aggregated features for {len(df)} users.")

    df = df.dropna()

    print(f"🎯 [INFO] Creating binary label using threshold: useful ≥ {useful_thres}")
    df['label'] = (df['useful'] >= useful_thres).astype(int)
    print(df.head(10))
    total = df['label'].shape[0]
    pos = (df['label'] == 1).sum()
    neg = (df['label'] == 0).sum()
    print(f"📊 [INFO] Label Distribution:")
    print(f"[INFO] ➤ Helpful     [1]: {pos} {pos / total:.4%}")
    print(f"[INFO] ➤ Not helpful [0]: {neg} {neg / total:.4%}")

    return df


def parse_hours_range(hr_str):
    try:
        open_hr, close_hr = map(str.strip, hr_str.split("-"))
        open_h, open_m = map(int, open_hr.split(":"))
        close_h, close_m = map(int, close_hr.split(":"))

        open_min = open_h * 60 + open_m
        close_min = close_h * 60 + close_m

        # handle overnight (close < open)
        if close_min <= open_min:
            close_min += 1440

        return round((close_min - open_min) / 60.0, 2)
    except:
        return np.nan


def business_feature(
        business_fp: str,
        checkin_fp: str,
        top_k: int,
        min_useful: int,
        useful_thres: float,
):
    print("📂 [INFO] Loading businesses...")
    df = pd.read_json(path_or_buf=business_fp, lines=True)
    print("📂 [INFO] Loading check-in...")
    checkin_df = pd.read_json(path_or_buf=checkin_fp, lines=True)
    checkin_df['checkin_count'] = len(checkin_df['date'])
    checkin_sum = checkin_df.groupby('business_id')['checkin_count'].sum()
    print("🔗 [INFO] Adding check-in to review...")
    df['check-in'] = df['business_id'].map(checkin_sum).fillna(0)

    features = [
        'business_id', 'stars', 'review_count', 'is_open',
        # 'attributes',
        'categories', 'hours', 'check-in',
    ]

    df = df[features].dropna()

    print("🧹 [INFO] Converting categories...")
    df['categories'] = df['categories'].str.split(', ')

    print(f"🧹 [INFO] Using top-{top_k} categories...")
    top_cats = df['categories'].explode().dropna().value_counts().head(top_k) \
        .index.tolist()
    print(f"📊 [INFO] Binarize category features...")
    for cat in top_cats:
        df[cat] = df['categories'].apply(lambda x: int(cat in x))

    print("🧹 [INFO] Converting features...")

    print("⏰ [INFO] Parsing business hours...")
    weekdays = [
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
        'Saturday', 'Sunday',
    ]
    rename_map = {
        'Monday': 'mon', 'Tuesday': 'tue', 'Wednesday': 'wed',
        'Thursday': 'thu', 'Friday': 'fri', 'Saturday': 'sat', 'Sunday': 'sun',
    }
    hours_df = df['hours'].dropna().apply(pd.Series)[weekdays].applymap(
        parse_hours_range
    )
    hours_df.rename(columns=rename_map, inplace=True)
    df = df.drop(columns='hours', errors='ignore')
    df = pd.concat(objs=[df, hours_df], axis=1)

    df = df.dropna()

    print(f"✅ [INFO] Final shape: {df.shape}")

    # print(df.columns)
    # df = df.select_dtypes(include='number')
    # print(df.columns)
    # print(df.columns)
    # for col in df.columns:
    #     print(df[col].nunique())
    #     print(df[col].dtype)

    return df


def binarize(
        df: DataFrame,
        method: Union[float, str],
):
    if type(method) == float:
        print(f"🔁 [INFO] Performing top {int((1 - method) * 100)}% binarization...")
    else:
        print(f"🔁 [INFO] Performing median binarization...")
    cols = df.select_dtypes(include='number')
    print("📊 [INFO] Distribution of 0s and 1s in binarized columns:")
    for col in cols:
        if col == 'label':
            continue

        if type(method) == float:
            thres = df[col].quantile(method)
        else:
            thres = df[col].median()

        df[col] = (df[col] >= thres).astype(int)
        total = len(df)
        ones = (df[col] == 1).sum()
        zeros = (df[col] == 0).sum()
        one_pct = ones / total
        zero_pct = zeros / total
        print(f"{col:<25}: 0s = {zeros:<6} [{zero_pct:8.4%}] | 1s = {ones:<6} [{one_pct:8.4%}]")

    print(f"✅ [INFO] Binarized {len(cols)} columns.")
    print(f"✅ [INFO] Final binarized user feature shape {df.shape}")

    return df


def draw_sankey_from_rules(rules, top_n=10):
    print("📊 [INFO] Drawing Sankey diagram...")

    # Prepare node labels
    rules = rules.sort_values(by='lift', ascending=False).head(top_n)
    antecedents = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
    consequents = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
    all_labels = list(set(antecedents) | set(consequents))
    label_idx = {label: i for i, label in enumerate(all_labels)}

    # Create source, target, value
    sources = antecedents.map(label_idx).tolist()
    targets = consequents.map(label_idx).tolist()
    values = rules['lift'].tolist()

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=all_labels, pad=20, thickness=20),
        link=dict(source=sources, target=targets, value=values)
    )])
    fig.update_layout(title_text="Sankey Diagram of Association Rules", font_size=12)
    fig.show()


def draw_category_network(itemsets_df, min_len=2, min_sup=0.01):
    print("📈 [INFO] Drawing category co-occurrence network...")

    G = nx.Graph()
    for _, row in itemsets_df.iterrows():
        items = list(row['itemsets'])
        support = row['support']
        if len(items) < min_len or support < min_sup:
            continue
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                if G.has_edge(items[i], items[j]):
                    G[items[i]][items[j]]['weight'] += support
                else:
                    G.add_edge(items[i], items[j], weight=support)

    pos = nx.spring_layout(G, seed=42)
    edge_weights = [G[u][v]['weight']*10 for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, width=edge_weights, node_size=1500, node_color='skyblue', font_size=10)
    plt.title("Category Co-occurrence Network")
    plt.show()


def draw_lift_support_heatmap(rules, top_n=20):
    # TODO: Try not to use Lift since it's null-variant !!! Use Kulczynski Measure
    print("📊 [INFO] Drawing heatmap of Lift vs. Support...")

    rules = rules.sort_values(by='lift', ascending=False).head(top_n)
    data = pd.DataFrame({
        'antecedent': rules['antecedents'].apply(lambda x: ', '.join(sorted(x))),
        'consequent': rules['consequents'].apply(lambda x: ', '.join(sorted(x))),
        'lift': rules['lift'],
    })

    pivot = data.pivot(index='antecedent', columns='consequent', values='lift').fillna(0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Lift Heatmap (Antecedents vs. Consequents)")
    plt.xlabel("Consequents")
    plt.ylabel("Antecedents")
    plt.tight_layout()
    plt.show()
