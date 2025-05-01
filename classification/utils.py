from pandas import DataFrame

import pandas as pd


def parse_elite(elite):
    if pd.isna(elite) or elite.strip() == '':
        return 0
    yrs = elite.replace("20,20", "2020").split(sep=', ')
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
    checkin_df['checkin_count'] = checkin_df['date'].str.count(",") + 1
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

    print("🎯 [INFO] Creating target variable...")
    df['label'] = (df['useful'] >= min_useful).astype(int)

    return df, top_cats
