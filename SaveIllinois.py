import pandas as pd
import os

# 定义数据路径 (请修改为你的数据集目录)
data_path = "data/"  # 原始数据路径
output_path = "yelp_illinois/"  # 用于存储 Illinois 数据的文件夹

# 如果不存在文件夹，则创建
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"已创建文件夹：{output_path}")

# Yelp 数据集文件
files = {
    "business": f"{data_path}/yelp_academic_dataset_business.json",
    "review": f"{data_path}/yelp_academic_dataset_review.json",
    "user": f"{data_path}/yelp_academic_dataset_user.json",
    "checkin": f"{data_path}/yelp_academic_dataset_checkin.json",
    "tip": f"{data_path}/yelp_academic_dataset_tip.json"
}

# 定义块大小
chunk_size = 500000  # 每次读取的行数

# 初始化商家 ID 集合和用户 ID 集合
illinois_business_ids = set()
illinois_user_ids = set()

# ---------------------------
# Step 1: 筛选 Illinois 商家
# ---------------------------
business_file = f"{output_path}/business_illinois.json"
total_processed = 0
total_saved = 0

if os.path.exists(business_file):
    # 从已有文件加载 business_id 集合
    business_df = pd.read_json(business_file, lines=True)
    illinois_business_ids.update(business_df['business_id'].tolist())
    print(f"已加载 {len(illinois_business_ids)} 个 Illinois 商家。")
else:
    print("正在筛选 Illinois 商家...")
    with pd.read_json(files["business"], lines=True, chunksize=chunk_size) as reader:
        for chunk in reader:
            total_processed += len(chunk)
            illinois_chunk = chunk[(chunk['state'] == 'IL') | (chunk['city'].str.contains('Chicago', case=False, na=False))]
            total_saved += len(illinois_chunk)
            illinois_business_ids.update(illinois_chunk['business_id'].tolist())
            illinois_chunk.to_json(business_file, orient='records', lines=True, mode='a')
            print(f"已处理 Business 数据 {total_processed} 条，已保存 Illinois 商家 {total_saved} 条。")
    print(f"Illinois 商家数量：{len(illinois_business_ids)}")


# ---------------------------
# Step 2: 提取 Review 数据
# ---------------------------
review_file = f"{output_path}/review_illinois.json"
total_processed = 0
total_saved = 0

print("正在筛选 Illinois 商家的所有评价...")
with pd.read_json(files["review"], lines=True, chunksize=chunk_size) as reader:
    for chunk in reader:
        total_processed += len(chunk)
        filtered_chunk = chunk[chunk['business_id'].isin(illinois_business_ids)]
        illinois_user_ids.update(filtered_chunk['user_id'].tolist())
        total_saved += len(filtered_chunk)
        filtered_chunk.to_json(review_file, orient='records', lines=True, mode='a')
        print(f"已处理 Review 数据 {total_processed} 条，已保存 Illinois 相关评论 {total_saved} 条。")
print(f"提取到的用户数量：{len(illinois_user_ids)}")


# ---------------------------
# Step 3: 提取 User 数据
# ---------------------------
user_file = f"{output_path}/user_illinois.json"
total_processed = 0
total_saved = 0

print("正在提取与 Illinois 商家相关的用户信息...")
with pd.read_json(files["user"], lines=True, chunksize=chunk_size) as reader:
    for chunk in reader:
        total_processed += len(chunk)
        filtered_chunk = chunk[chunk['user_id'].isin(illinois_user_ids)]
        total_saved += len(filtered_chunk)
        filtered_chunk.to_json(user_file, orient='records', lines=True, mode='a')
        print(f"已处理 User 数据 {total_processed} 条，已保存 Illinois 相关用户 {total_saved} 条。")
print("用户数据提取完成。")


# ---------------------------
# Step 4: 提取 Check-in 数据
# ---------------------------
checkin_file = f"{output_path}/checkin_illinois.json"
total_processed = 0
total_saved = 0

print("正在提取与 Illinois 商家相关的 Check-in 信息...")
with pd.read_json(files["checkin"], lines=True, chunksize=chunk_size) as reader:
    for chunk in reader:
        total_processed += len(chunk)
        filtered_chunk = chunk[chunk['business_id'].isin(illinois_business_ids)]
        total_saved += len(filtered_chunk)
        filtered_chunk.to_json(checkin_file, orient='records', lines=True, mode='a')
        print(f"已处理 Check-in 数据 {total_processed} 条，已保存 Illinois 相关 Check-in {total_saved} 条。")
print("Check-in 数据提取完成。")


# ---------------------------
# Step 5: 提取 Tip 数据
# ---------------------------
tip_file = f"{output_path}/tip_illinois.json"
total_processed = 0
total_saved = 0

print("正在提取与 Illinois 商家相关的 Tip 信息...")
with pd.read_json(files["tip"], lines=True, chunksize=chunk_size) as reader:
    for chunk in reader:
        total_processed += len(chunk)
        filtered_chunk = chunk[chunk['business_id'].isin(illinois_business_ids)]
        total_saved += len(filtered_chunk)
        filtered_chunk.to_json(tip_file, orient='records', lines=True, mode='a')
        print(f"已处理 Tip 数据 {total_processed} 条，已保存 Illinois 相关 Tip {total_saved} 条。")
print("Tip 数据提取完成。")

print("\n所有 Illinois 数据提取完成！")
