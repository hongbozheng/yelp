from typing import List
from pandas import DataFrame

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


def merge_df(
        business_fp: str,
        review_fp: str,
        user_fp: str,
):
    print("🔄 Loading filtered datasets...")
    business_df = pd.read_json(business_fp, lines=True)
    review_df = pd.read_json(review_fp, lines=True)
    user_df = pd.read_json(user_fp, lines=True)

    print("🔗 Merging reviews with users...")
    merged_df = review_df.merge(
        user_df[['user_id', 'review_count', 'average_stars', 'fans']],
        on='user_id', how='inner'
    )

    print("🔗 Merging with business metadata...")
    merged_df = merged_df.merge(
        business_df[['business_id', 'categories', 'stars', 'name', 'city']],
        on='business_id', how='inner'
    )

    print(f"✅ Final merged shape: {merged_df.shape}")
    return merged_df


def preprocess_data(
        dir: str,
        start_date: str,
        min_review: int,
):
    business_fp = os.path.join(dir, 'yelp_academic_dataset_business.json')
    review_fp = os.path.join(dir, 'yelp_academic_dataset_review.json')
    user_fp = os.path.join(dir, 'yelp_academic_dataset_user.json')

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
    print(f"✅ [INFO] {len(review_df)} reviews retained.")

    user_ids = set(review_df['user_id'])

    print("👤 [INFO] Step 4 Filter users...")
    user_chunks = []
    reader = pd.read_json(user_fp, lines=True, chunksize=100000)
    for chunk in tqdm(iterable=reader, desc="[INFO] Filtering users"):
        # chunk['review_count'] = pd.to_numeric(
        #     chunk['review_count'], errors='coerce'
        # )
        chunk = chunk[
            chunk['user_id'].isin(user_ids) &
            (chunk['review_count'] >= min_review)
        ]
        user_chunks.append(chunk)
    user_df = pd.concat(user_chunks, ignore_index=True)
    print(f"✅ [INFO] {len(user_df)} users retained.")

    print("💾 [INFO] Step 5 Save filtered datasets...")
    user_ids = set(user_df['user_id'])
    review_df = review_df[review_df['user_id'].isin(user_ids)]

    print("💾 [INFO] Step 5 Save filtered datasets...")
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
    print("🎉 [INFO] Done! All filtered datasets saved.")

    print("🔗 [INFO] Merging review + user...")
    review_df = review_df.merge(
        user_df, on="user_id", how="inner", suffixes=('', '_user')
    )

    print("🔗 [INFO] Merging with business...")
    review_df = review_df.merge(
        business_df, on="business_id", how="inner", suffixes=('', '_biz')
    )

    review_df.to_json(
        path_or_buf=f"{dir}/dataset.json",
        orient="records",
        lines=True,
    )
    print(f"✅ [INFO] Final merged shape: {review_df.shape}")

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

    # shape [595787, 43]
    preprocess_data(dir=dir, start_date=start_date, min_review=min_review)
