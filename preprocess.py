from typing import List

import argparse
import os
import pandas as pd
from collections import Counter
from tqdm import tqdm


def get_top_review_cities(
        business_path: str,
        review_path: str,
        top_n: int = 10,
) -> List[str]:
    print("🔍 [INFO] Loading business metadata...")
    business_df = pd.read_json(
        path_or_buf=business_path,
        lines=True,
        dtype={'business_id': str, 'city': str},
    )
    biz_city = business_df.set_index('business_id')['city'].to_dict()

    print("🔄 [INFO] Scanning review dataset...")
    city_counter = Counter()
    chunks = pd.read_json(path_or_buf=review_path, lines=True, chunksize=100000)

    for chunk in tqdm(iterable=chunks, desc="[INFO] Processing review chunks"):
        chunk['city'] = chunk['business_id'].map(biz_city)
        chunk = chunk.dropna(subset=['city'])
        city_counter.update(chunk['city'])

    top_cities = city_counter.most_common(top_n)

    print("📊 [INFO] Top Cities by Review Count:")
    for city, count in top_cities:
        print(f"[INFO] {city}: {count:,} reviews")

    top_cities = [city for city, _ in top_cities]
    print("✅ [INFO] Top City List for Filtering:", top_cities)

    return top_cities


def filter_checkin(date_str, date):
    if not date_str:
        return []
    try:
        dates = pd.to_datetime(
            date_str.split(', '), format='%Y-%m-%d %H:%M:%S', errors='coerce'
        )
        filtered = [
            d.strftime('%Y-%m-%d %H:%M:%S') for d in dates if d and d >= date
        ]
        return filtered
    except Exception:
        return []


def preprocess(
        dir: str,
        start_date: str,
        min_review: int,
):
    business_fp = os.path.join(dir, 'yelp_academic_dataset_business.json')
    review_fp = os.path.join(dir, 'yelp_academic_dataset_review.json')
    user_fp = os.path.join(dir, 'yelp_academic_dataset_user.json')
    checkin_fp = os.path.join(dir, 'yelp_academic_dataset_checkin.json')
    tip_fp = os.path.join(dir, 'yelp_academic_dataset_tip.json')

    print("📊 [INFO] Step 1 Get top review cities...")
    # top_cities = get_top_review_cities(
    #     business_fp=business_fp,
    #     review_fp=review_fp,
    #     top_n=10,
    # )
    top_cities = [
        'Philadelphia',
        'New Orleans',
        'Tampa',
        'Nashville',
        'Tucson',
        'Indianapolis',
        'Reno',
        'Santa Barbara',
        'Saint Louis',
        'Boise',
    ]

    print("🏢 [INFO] Step 2 Filter businesses...")
    business_df = pd.read_json(
        path_or_buf=business_fp,
        lines=True,
        dtype={'business_id': str, 'city': str},
    )
    business_df = business_df[business_df['city'].isin(top_cities)]
    business_ids = set(business_df['business_id'])
    print(f"✅ [INFO] {len(business_df)} businesses retained.")

    print("📝 [INFO] Step 3 Filter reviews...")
    review_chunks = []
    reader = pd.read_json(path_or_buf=review_fp, lines=True, chunksize=100000)
    for chunk in tqdm(iterable=reader, desc="[INFO] Filtering reviews"):
        chunk = chunk[
            chunk['business_id'].isin(business_ids) &
            (pd.to_datetime(chunk['date']) >= pd.to_datetime(start_date))
        ]
        review_chunks.append(chunk)
    review_df = pd.concat(objs=review_chunks, ignore_index=True)

    user_review_counts = review_df['user_id'].value_counts()
    user_ids = user_review_counts[user_review_counts >= min_review].index
    review_df = review_df[review_df['user_id'].isin(user_ids)]
    print(f"✅ [INFO] {len(user_ids)} users have ≥{min_review} relevant reviews.")
    print(f"✅ [INFO] {len(review_df)} reviews retained.")

    print("👤 [INFO] Step 4 Filter users...")
    user_chunks = []
    reader = pd.read_json(user_fp, lines=True, chunksize=100000)
    for chunk in tqdm(iterable=reader, desc="[INFO] Filtering users"):
        chunk = chunk[chunk['user_id'].isin(user_ids)]
        user_chunks.append(chunk)
    user_df = pd.concat(user_chunks, ignore_index=True)
    print(f"✅ [INFO] {len(user_df)} users retained.")

    print("📅 Step 5 Filter check-ins...")
    checkin_chunks = []
    date = pd.to_datetime(start_date)
    reader = pd.read_json(checkin_fp, lines=True, chunksize=100000)
    for chunk in tqdm(iterable=reader, desc="[INFO] Filtering check-ins"):
        chunk = chunk[chunk['business_id'].isin(business_ids)]
        chunk = chunk.copy()
        chunk.loc[:, 'date'] = chunk['date'].apply(
            lambda x: filter_checkin(x, date)
        )
        checkin_chunks.append(chunk)
    checkin_df = pd.concat(checkin_chunks, ignore_index=True)
    print(f"✅ [INFO] {len(checkin_df)} check-ins retained.")

    print("📅 Step 6 Filter tip...")
    tip_chunks = []
    reader = pd.read_json(tip_fp, lines=True, chunksize=100000)
    for chunk in tqdm(iterable=reader, desc="[INFO] Filtering tips"):
        chunk = chunk[
            chunk['user_id'].isin(user_ids) &
            chunk['business_id'].isin(business_ids)
        ].copy()
        chunk['tip_len'] = chunk['text'].str.len()
        chunk['date'] = pd.to_datetime(chunk['date'])
        tip_chunks.append(chunk)
    tip_df = pd.concat(tip_chunks, ignore_index=True)
    tip_df = tip_df.groupby(['user_id', 'business_id']).agg(
        tip_count=('text', 'count'),
        avg_tip_len=('tip_len', 'mean'),
        last_tip_date=('date', 'max'),
        compliment_count=('compliment_count', 'sum'),
    ).reset_index()
    tip_df['last_tip_date'] = tip_df['last_tip_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    print(f"✅ [INFO] {len(tip_df)} tip retained.")

    print("💾 [INFO] Step 7 Save filtered datasets...")
    business_df.to_json(
        path_or_buf=f"{dir}/business.json",
        orient="records",
        lines=True,
    )
    review_df.to_json(
        path_or_buf=f"{dir}/review.json",
        orient="records",
        lines=True,
    )
    user_df.to_json(
        path_or_buf=f"{dir}/user.json",
        orient="records",
        lines=True,
    )
    checkin_df.to_json(
        path_or_buf=f"{dir}/checkin.json",
        orient="records",
        lines=True,
    )
    tip_df.to_json(
        path_or_buf=f"{dir}/tip.json",
        orient="records",
        lines=True,
    )
    print("🎉 [INFO] Done! All filtered datasets saved.")

    # print("🔗 [INFO] Merging review with business...")
    # review_df = review_df.merge(
    #     business_df, on="business_id", how="inner", suffixes=('', '_biz')
    # )
    #
    # print("🔗 [INFO] Merging with check-in...")
    # review_df = review_df.merge(
    #     checkin_df, on='business_id', how='left', suffixes=('', '_checkin')
    # )
    #
    # print("🔗 [INFO] Merging with tip...")
    # review_df = review_df.merge(
    #     tip_df,
    #     on=['user_id', 'business_id'],
    #     how='left',
    #     suffixes=('', '_tip'),
    # )
    #
    # review_df.to_json(
    #     path_or_buf=f"{dir}/dataset.json",
    #     orient="records",
    #     lines=True,
    # )
    # print(f"✅ [INFO] Final merged shape: {review_df.shape}")
    #
    # print("📋 [INFO] Number of missing values per column:")
    # missing_counts = review_df.isna().sum()
    # print(missing_counts[missing_counts > 0].sort_values(ascending=False))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="preprocess.py", description="Preprocess the Yelp dataset"
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        required=True,
        help="Directory of the Yelp dataset",
    )
    parser.add_argument(
        "--start_date",
        "-s",
        type=str,
        required=True,
        help="Start date to filter the Yelp dataset",
    )
    parser.add_argument(
        "--min_review",
        "-m",
        type=int,
        required=True,
        help="Minimum number of reviews",
    )
    args = parser.parse_args()
    dir = args.directory
    start_date = args.start_date
    min_review = args.min_review

    # shape [206130, 27]
    preprocess(dir=dir, start_date=start_date, min_review=min_review)
