import json
import os
import pandas as pd

# name = ['jacket', 'skirt', 'trousers']
source_dir = r"C:\Users\lenovo\Desktop\demo\dataset\ATVI"
name = [n for n in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, n))]

baselines = ['Prom']
# baselines = ['GA', 'GSP', 'GSP_SQA', 'Prom_ALS', 'Prom_AAL']
llms = ['ChatGPT', 'claude', 'ernie', 'GLM', 'kimi', 'Qwen']

for baseline in baselines:
    for i in range(10):
        for j in range(14):
            ad_sw, ad_rev, ad_utility, user_satis, ad_satis, score = [[] for _ in range(6)]
            llm_ls = []
            for llm in llms:
                file_path = (f'C:/Users/lenovo/Desktop/demo/tasks_atvi/s-t2/result_{baseline}/{name[j]}_section{i + 1}/'
                             f'{llm}_results_{name[j]}_section{i + 1}.json')
                if os.path.exists(file_path):
                    llm_ls.append(llm)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        ad_sw.append(data['ad_sw'])
                        ad_rev.append(sum(data['ad_rev']))
                        ad_utility.append(sum(data['ad_utility']))
                        user_satis.append(data['user_satis'])
                        ad_satis.append(data['ad_satis'])
                        score.append(data['score'])

            output_dir = f'C:/Users/lenovo/Desktop/demo/tasks_atvi/s-t2/result_{baseline}/{name[j]}_section{i + 1}'
            if os.path.exists(output_dir):
                table_data = {
                    "ad_sw": ad_sw,
                    "ad_rev_sum": ad_rev,
                    "ad_utility_sum": ad_utility,
                    "user_satis": user_satis,
                    "ad_satis": ad_satis,
                    "score": score
                }
                df = pd.DataFrame(table_data, index=llm_ls).transpose()
                max_score_col = df.loc['score'].idxmax()
                df['max_score'] = df[max_score_col]

                # df['mean'] = df.loc[["ad_sw", "ad_rev_sum", "ad_utility_sum", "user_satis", "ad_satis"]].mean(axis=1)
                # df.loc['score', 'mean'] = df.loc['score'].max()

                # df['mean'] = df.loc[["ad_sw", "ad_rev_sum", "ad_utility_sum", "user_satis", "ad_satis", 'score']].mean(
                #     axis=1)

                output_file = os.path.join(output_dir, f'{name[j]}_section{i + 1}_table.xlsx')
                df.to_excel(output_file, index=True)

    all_means = []
    for i in range(10):
        for j in range(14):
            table_path = (f'C:/Users/lenovo/Desktop/demo/tasks_atvi/s-t2/result_{baseline}/'
                          f'{name[j]}_section{i + 1}/{name[j]}_section{i + 1}_table.xlsx')

            if os.path.exists(table_path):
                df = pd.read_excel(table_path, index_col=0)
                last_column_means = df.iloc[:, -1]
                all_means.append(last_column_means)

    df_means = pd.DataFrame(all_means)
    final_mean = df_means.mean()
    final_var = df_means.var()
    print(final_mean)
    print(final_var)
