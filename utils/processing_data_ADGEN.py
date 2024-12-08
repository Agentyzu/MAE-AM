import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os

# 创建一个默认字典来统计每种类型的数量
type_count = defaultdict(int)
type_summary_lengths = defaultdict(list)
type_summaries = defaultdict(list)

# 逐行读取.json文件
ADGEN_path = 'C:/Users/lenovo/Desktop/demo/dataset/ADGEN/dev.json'
ad_name_path = 'C:/Users/lenovo/Desktop/demo/dataset/ADGEN/ad_name.txt'

with open(ADGEN_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        # 提取content中的类型部分
        content = data.get('content', '')
        summary = data.get('summary', '')

        is_synthetic = True

        # 假设类型总是以"类型#"开头并且用"*"分隔
        type_info = content.split('*')[0]
        if "类型#" in type_info:
            # 获取"类型"的值
            item_type = type_info.split('#')[1]

            # 统计该类型的出现次数
            type_count[item_type] += 1

            # 统计summary的长度
            summary_length = len(summary)
            type_summary_lengths[item_type].append(summary_length)

            # 记录summary和其长度
            type_summaries[item_type].append((summary, len(summary)))

with open(ad_name_path, 'r', encoding='utf-8') as f:
    ad_names = [line.strip() for line in f.readlines()]

# 输出有多少种类型，以及每个类型中有多少条信息
print(f"总共有 {len(type_count)} 种类型:")
for item_type, count in type_count.items():
    print(f"类型: {item_type}, 条数: {count}")

# 为每个类型单独画图
dic = {"裤": "trousers", "裙": "skirt", "上衣": "jacket"}
for item_type, lengths in type_summary_lengths.items():
    plt.figure(figsize=(8, 6))
    plt.hist(lengths, bins=3, edgecolor='black')
    plt.title(f'Summary Length Distribution for Type {dic[item_type]}')
    plt.xlabel('Summary Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

for item_type, summaries_and_lengths in type_summaries.items():
    folder_name = f'C:/Users/lenovo/Desktop/demo/dataset/ADGEN_100/{dic[item_type]}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 将长度分成若干区间
    lengths = [length for _, length in summaries_and_lengths]
    bins = np.histogram_bin_edges(lengths, bins=3)  # 自动生成10个区间

    # 遍历每个区间，收集每个区间的summary
    for i in range(len(bins) - 1):
        # 找出当前区间内的所有summary
        current_bin_summaries = [summary for summary, length in summaries_and_lengths
                                 if bins[i] <= length < bins[i + 1]]

        # if len(current_bin_summaries) < 10:
        #     continue

        # 从中随机采样20个（如果少于20个，全部选出）
        sample_count = min(100, len(current_bin_summaries))
        sampled_summaries = random.sample(current_bin_summaries, sample_count)


        # 保存到JSON文件，命名为"类型_区间{idx}.json"
        output_data = []
        for idx, content in enumerate(sampled_summaries):
            if is_synthetic:  # 合成数据时随机生成0-1的点击率
                pctr = random.uniform(0.3, 0.5)
                ad_name = random.choice(ad_names)
                output_data.append({"ad_id": idx, "ad_name": ad_name, "pctr": pctr, "content": content})

        output_filename = os.path.join(folder_name, f"{dic[item_type]}_section{i + 1}.json")

        # 写入文件
        with open(output_filename, 'w', encoding='utf-8') as json_file:
            json.dump(output_data, json_file, ensure_ascii=False, indent=4)
