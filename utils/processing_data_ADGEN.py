import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os

# Create a default dictionary to count the occurrence of each type
type_count = defaultdict(int)
type_summary_lengths = defaultdict(list)
type_summaries = defaultdict(list)

# Paths to the input JSON file and the ad names text file
ADGEN_path = 'dataset/ADGEN/dev.json'
ad_name_path = 'dataset/ADGEN/ad_name.txt'

# Read the JSON file line by line and process the content
with open(ADGEN_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)  # Parse the line as a JSON object
        content = data.get('content', '')  # Get the content field
        summary = data.get('summary', '')  # Get the summary field

        is_synthetic = True  # Flag to indicate synthetic data

        # Assume the type is always at the beginning of the content and delimited by '*'
        type_info = content.split('*')[0]
        if "类型#" in type_info:
            # Extract the type value after '类型#'
            item_type = type_info.split('#')[1]

            # Increment the count for the given type
            type_count[item_type] += 1

            # Track the length of each summary
            summary_length = len(summary)
            type_summary_lengths[item_type].append(summary_length)

            # Store the summary along with its length
            type_summaries[item_type].append((summary, len(summary)))

# Read the ad names from the text file
with open(ad_name_path, 'r', encoding='utf-8') as f:
    ad_names = [line.strip() for line in f.readlines()]

# Output the total number of types and the number of entries for each type
print(f"Total types: {len(type_count)}")
for item_type, count in type_count.items():
    print(f"Type: {item_type}, Count: {count}")

# Dictionary to map item types to their English names
dic = {"裤": "trousers", "裙": "skirt", "上衣": "jacket"}

# Generate histograms for each type to visualize summary length distribution
for item_type, lengths in type_summary_lengths.items():
    plt.figure(figsize=(8, 6))  # Set figure size
    plt.hist(lengths, bins=3, edgecolor='black')  # Create histogram with 3 bins
    plt.title(f'Summary Length Distribution for Type {dic[item_type]}')  # Title of the plot
    plt.xlabel('Summary Length')  # X-axis label
    plt.ylabel('Frequency')  # Y-axis label
    plt.grid(True)  # Show grid
    plt.show()  # Display the plot

# Process summaries by length bins and sample them randomly
for item_type, summaries_and_lengths in type_summaries.items():
    # Create folder for each type if it doesn't exist
    folder_name = f'dataset/ADGEN_100/{dic[item_type]}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Extract the summary lengths
    lengths = [length for _, length in summaries_and_lengths]

    # Get bin edges for histogram (auto generate 3 bins)
    bins = np.histogram_bin_edges(lengths, bins=3)

    # Iterate through each bin and collect summaries that fall into the bin
    for i in range(len(bins) - 1):
        # Find summaries that fall within the current bin's length range
        current_bin_summaries = [summary for summary, length in summaries_and_lengths
                                 if bins[i] <= length < bins[i + 1]]

        # If there are fewer than 10 summaries in the bin, skip it
        # if len(current_bin_summaries) < 10:
        #     continue

        # Sample 100 summaries from the bin (if there are fewer than 100, sample all)
        sample_count = min(100, len(current_bin_summaries))
        sampled_summaries = random.sample(current_bin_summaries, sample_count)

        # Prepare data for output
        output_data = []
        for idx, content in enumerate(sampled_summaries):
            if is_synthetic:  # If synthetic data, generate random click-through rate
                pctr = random.uniform(0.3, 0.5)
                ad_name = random.choice(ad_names)
                output_data.append({"ad_id": idx, "ad_name": ad_name, "pctr": pctr, "content": content})

        # Output filename with type and bin section information
        output_filename = os.path.join(folder_name, f"{dic[item_type]}_section{i + 1}.json")

        # Write the output data to a JSON file
        with open(output_filename, 'w', encoding='utf-8') as json_file:
            json.dump(output_data, json_file, ensure_ascii=False, indent=4)
