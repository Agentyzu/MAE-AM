import pandas as pd
import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os

ATVI_path = 'dataset/ATVI/ad_data.xlsx'
data = pd.read_excel(ATVI_path)

category_count = defaultdict(int)
category_summary_lengths = defaultdict(list)
category_summaries = defaultdict(list)

for index, row in data.iterrows():
    category = row['Category']
    ad_copy = row['Ad_copy']
    advertiser = row['Advertiser']

    category_count[category] += 1

    ad_copy_length = len(ad_copy)
    category_summary_lengths[category].append(ad_copy_length)

    category_summaries[category].append((ad_copy, ad_copy_length, advertiser))

for category, lengths in category_summary_lengths.items():
    plt.figure(figsize=(8, 6))
    plt.hist(lengths, bins=10, edgecolor='black')
    plt.title(f'Summary Length Distribution for Category {category}')
    plt.xlabel('Ad Copy Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

for category, summaries_and_lengths in category_summaries.items():
    folder_name = f'dataset/ATVI/{category}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    lengths = [length for _, length, _ in summaries_and_lengths]
    bins = np.histogram_bin_edges(lengths, bins=10)

    for i in range(len(bins) - 1):
        current_bin_summaries = [(ad_copy, advertiser) for ad_copy, length, advertiser in summaries_and_lengths
                                 if bins[i] <= length < bins[i + 1]]

        if len(current_bin_summaries) < 10:
            continue

        sample_count = min(20, len(current_bin_summaries))
        sampled_summaries = random.sample(current_bin_summaries, sample_count)

        output_data = []
        for idx, (content, advertiser) in enumerate(sampled_summaries):
            pctr = random.uniform(0.3, 0.5)
            output_data.append({"ad_id": idx, "ad_name": advertiser, "pctr": pctr, "content": content})

        output_filename = os.path.join(folder_name, f"{category}_section{i + 1}.json")

        with open(output_filename, 'w', encoding='utf-8') as json_file:
            json.dump(output_data, json_file, ensure_ascii=False, indent=4)
