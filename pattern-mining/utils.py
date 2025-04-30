from pandas import DataFrame

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns


def load_merge(review_fp: str, business_fp: str) -> DataFrame:
    print("📂 [INFO] Loading reviews...")
    review_df = pd.read_json(path_or_buf=review_fp, lines=True)
    print("📂 [INFO] Loading businesses...")
    business_df = pd.read_json(path_or_buf=business_fp, lines=True)

    print("🧹 [INFO] Splitting and cleaning categories...")
    business_df = business_df[['business_id', 'city', 'categories']].dropna()
    business_df['categories'] = business_df['categories'].str.split(', ')
    print("🔗 [INFO] Merging to attach city and categories...")
    df = review_df.merge(business_df, on='business_id', how='inner')

    return df


def one_hot_encode(df: DataFrame, top_k: int = 50) -> DataFrame:
    print("📊 [INFO] One-hot encoding categories...")
    all_cats = df['categories'].explode().dropna()
    top_cats = all_cats.value_counts().head(top_k).index.tolist()
    print(f"✅ [INFO] Using top {top_k} categories.")

    df['filtered_cats'] = df['categories'].apply(
        lambda x: list(set(x).intersection(top_cats))
    )

    ohe_df = pd.DataFrame(data=False, index=df.index, columns=top_cats)
    for i, cats in df['filtered_cats'].items():
        for cat in cats:
            ohe_df.at[i, cat] = True

    return ohe_df


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
