import pandas as pd
import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取Excel文件
ATVI_path = 'C:/Users/lenovo/Desktop/demo/dataset/ATVI/ad_data.xlsx'
data = pd.read_excel(ATVI_path)

# 创建一个默认字典来统计每种类型的数量
category_count = defaultdict(int)
category_summary_lengths = defaultdict(list)
category_summaries = defaultdict(list)

# 逐行读取Excel数据
for index, row in data.iterrows():
    category = row['Category']
    ad_copy = row['Ad_copy']
    advertiser = row['Advertiser']  # 获取广告商名称

    # 统计该类型的出现次数
    category_count[category] += 1

    # 统计ad_copy的长度
    ad_copy_length = len(ad_copy)
    category_summary_lengths[category].append(ad_copy_length)

    # 记录ad_copy、长度和广告商
    category_summaries[category].append((ad_copy, ad_copy_length, advertiser))

# 输出有多少种类型，以及每个类型中有多少条广告信息
print(f"总共有 {len(category_count)} 种广告类型:")
for category, count in category_count.items():
    print(f"广告类型: {category}, 条数: {count}")

# 为每个类型单独画图
for category, lengths in category_summary_lengths.items():
    plt.figure(figsize=(8, 6))
    plt.hist(lengths, bins=10, edgecolor='black')
    plt.title(f'Summary Length Distribution for Category {category}')
    plt.xlabel('Ad Copy Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# 创建文件夹并保存区间划分后的JSON文件
for category, summaries_and_lengths in category_summaries.items():
    folder_name = f'C:/Users/lenovo/Desktop/demo/dataset/ATVI/{category}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 将长度分成若干区间
    lengths = [length for _, length, _ in summaries_and_lengths]
    bins = np.histogram_bin_edges(lengths, bins=10)  # 自动生成10个区间

    # 遍历每个区间，收集每个区间的广告文案
    for i in range(len(bins) - 1):
        # 找出当前区间内的所有广告文案
        current_bin_summaries = [(ad_copy, advertiser) for ad_copy, length, advertiser in summaries_and_lengths
                                 if bins[i] <= length < bins[i + 1]]

        if len(current_bin_summaries) < 10:
            continue

        # 从中随机采样20个（如果少于20个，全部选出）
        sample_count = min(20, len(current_bin_summaries))
        sampled_summaries = random.sample(current_bin_summaries, sample_count)

        # 保存到JSON文件
        output_data = []
        for idx, (content, advertiser) in enumerate(sampled_summaries):
            # 随机生成点击率pctr
            pctr = random.uniform(0.3, 0.5)
            output_data.append({"ad_id": idx, "ad_name": advertiser, "pctr": pctr, "content": content})

        output_filename = os.path.join(folder_name, f"{category}_section{i + 1}.json")

        # 写入文件
        with open(output_filename, 'w', encoding='utf-8') as json_file:
            json.dump(output_data, json_file, ensure_ascii=False, indent=4)
