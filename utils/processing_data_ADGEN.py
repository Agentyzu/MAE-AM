import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os

type_count = defaultdict(int)
type_summary_lengths = defaultdict(list)
type_summaries = defaultdict(list)

ADGEN_path = 'dataset/ADGEN/dev.json'
ad_name_path = 'dataset/ADGEN/ad_name.txt'

with open(ADGEN_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        content = data.get('content', '')
        summary = data.get('summary', '')

        is_synthetic = True

        type_info = content.split('*')[0]
        if "类型#" in type_info:
            item_type = type_info.split('#')[1]

            type_count[item_type] += 1

            summary_length = len(summary)
            type_summary_lengths[item_type].append(summary_length)

            type_summaries[item_type].append((summary, len(summary)))

with open(ad_name_path, 'r', encoding='utf-8') as f:
    ad_names = [line.strip() for line in f.readlines()]

print(f"Total types: {len(type_count)}")
for item_type, count in type_count.items():
    print(f"Type: {item_type}, Count: {count}")

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
    folder_name = f'dataset/ADGEN_100/{dic[item_type]}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    lengths = [length for _, length in summaries_and_lengths]

    bins = np.histogram_bin_edges(lengths, bins=3)

    for i in range(len(bins) - 1):
        current_bin_summaries = [summary for summary, length in summaries_and_lengths
                                 if bins[i] <= length < bins[i + 1]]


        sample_count = min(100, len(current_bin_summaries))
        sampled_summaries = random.sample(current_bin_summaries, sample_count)

        output_data = []
        for idx, content in enumerate(sampled_summaries):
            if is_synthetic:
                pctr = random.uniform(0.3, 0.5)
                ad_name = random.choice(ad_names)
                output_data.append({"ad_id": idx, "ad_name": ad_name, "pctr": pctr, "content": content})

        output_filename = os.path.join(folder_name, f"{dic[item_type]}_section{i + 1}.json")

        with open(output_filename, 'w', encoding='utf-8') as json_file:
            json.dump(output_data, json_file, ensure_ascii=False, indent=4)
