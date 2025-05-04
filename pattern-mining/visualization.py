#!/usr/bin/env python3
# visualize_helpfulness.py

import os
import sys
import argparse
import pandas as pd
import ast
import re
import plotly.express as px
from plotly.subplots import make_subplots

def parse_itemset(s: str):
    """
    Turn strings like "frozenset({'A','B'})" or "{'A','B'}" into a Python set.
    """
    m = re.match(r"frozenset\((.*)\)", s.strip())
    inner = m.group(1) if m else s
    return set(ast.literal_eval(inner))

def load_and_process(path, diff_col):
    df = pd.read_csv(path)
    df['itemset'] = df['itemsets'].apply(parse_itemset)
    df['pattern'] = df['itemset'].apply(lambda s: ", ".join(sorted(s)))
    return df.sort_values(diff_col, ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize exclusive patterns for helpful vs. non-helpful reviews"
    )
    parser.add_argument(
        "--task", choices=["user","category"], default="user",
        help="Which feature set to visualize"
    )
    parser.add_argument(
        "--n", "-n", type=int, default=10,
        help="Number of top patterns to show"
    )
    args = parser.parse_args()

    help_file = f"results/{args.task}-exclusive-helpful.csv"
    unh_file  = f"results/{args.task}-exclusive-unhelpful.csv"
    for f in (help_file, unh_file):
        if not os.path.exists(f):
            sys.exit(f"ERROR: missing file {f}")

    # Determine which diff column is present
    sample = pd.read_csv(help_file, nrows=1)
    diff_col = 'count_diff' if 'count_diff' in sample.columns else 'support_diff'

    df_help = load_and_process(help_file, diff_col)
    df_unh  = load_and_process(unh_file,  diff_col)

    topn = args.n
    df_hn = df_help.head(topn)
    df_un = df_unh.head(topn)

    # Create a 1x2 subplot figure
    fig = make_subplots(
        rows=1, cols=1,
        shared_yaxes=True,
        horizontal_spacing=0.0,
        subplot_titles=["Exclusive to Helpful", "Exclusive to Helpful"]
    )

    # Left: Helpful
    bar_h = px.bar(
        df_hn,
        x=diff_col,
        y="pattern",
        orientation='h'
    )
    for trace in bar_h.data:
        fig.add_trace(trace, row=1, col=1)

    if args.task == "category":
        # Right: Unhelpful (sort so most negative at top)
        bar_u = px.bar(
            df_un,
            x=diff_col,
            y="pattern",
            orientation='h'
        )
        for trace in bar_u.data:
            fig.add_trace(trace, row=1, col=1)

    # Update layout
    fig.update_layout(
        height=600,
        width=1000,
        title_text="Top Exclusive Patterns: Helpful vs. Unhelpful",
        showlegend=False
    )
    # Reverse y-axis so top bar is largest magnitude
    fig.update_yaxes(autorange="reversed")

    # Write to standalone HTML and open
    outpath = "results/patterns_exclusive.html"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.write_html(
        file=outpath,
        include_plotlyjs="cdn",
        full_html=True,
        auto_open=True
    )
    print(f"🌐 [INFO] Visualization written to {outpath} and opened in browser.")
