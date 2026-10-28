import argparse
import json
import os
import pandas as pd

name = ['jacket', 'skirt', 'trousers']

baselines = ['Prom']
llms = ['ChatGPT', 'claude', 'ernie', 'GLM', 'kimi', 'Qwen']

result_root = os.environ.get('MAE_AM_RESULT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sections = 10
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', default=result_root, help='Directory containing result_<baseline> folders')
    parser.add_argument('--categories', nargs='+', default=name)
    parser.add_argument('--baselines', nargs='+', default=baselines)
    parser.add_argument('--models', nargs='+', default=llms)
    parser.add_argument('--sections', type=int, default=sections)
    args = parser.parse_args()
    if args.sections < 1:
        parser.error('--sections must be positive')
    result_root, name, baselines, llms, sections = (
        args.result_root, args.categories, args.baselines, args.models, args.sections)
result_root = os.path.abspath(os.path.expanduser(result_root))

for baseline in baselines:
    processed_tables = []
    missing_groups = []
    incomplete_groups = []
    for i in range(sections):
        for j in range(len(name)):
            group = f'{name[j]}_section{i + 1}'
            output_dir = os.path.join(result_root, f'result_{baseline}', group)
            ad_sw, ad_rev, ad_utility, user_satis, ad_satis, score, model_mean_scores = [[] for _ in range(7)]
            llm_ls = []
            candidate_order = []
            for llm in llms:
                file_path = os.path.join(output_dir, f'{llm}_results_{group}.json')
                if os.path.exists(file_path):
                    llm_ls.append(llm)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        model_mean_scores.append(data.get('mean_score', data['score']))
                        declared = data.get('candidate_models', [])
                        if declared:
                            if candidate_order and declared != candidate_order:
                                raise ValueError(f'Inconsistent candidate model order: {file_path}')
                            candidate_order = declared
                        if 'samples' in data:
                            data = {**data, **data['samples'][data['best_sample_index']]}
                        ad_sw.append(data['ad_sw'])
                        ad_rev.append(sum(data['ad_rev']))
                        ad_utility.append(sum(data['ad_utility']))
                        user_satis.append(data['user_satis'])
                        ad_satis.append(data['ad_satis'])
                        score.append(data['score'])

            if not llm_ls:
                missing_groups.append(group)
                continue
            missing_models = [m for m in (candidate_order or llms) if m not in llm_ls]
            if missing_models:
                incomplete_groups.append((group, missing_models))
                print(f'Skip {baseline}/{group}: missing model logs {missing_models}')
                continue
            if os.path.exists(output_dir):
                table_data = {
                    "ad_sw": ad_sw,
                    "ad_rev_sum": ad_rev,
                    "ad_utility_sum": ad_utility,
                    "user_satis": user_satis,
                    "ad_satis": ad_satis,
                    "score": score,
                    "model_mean_score": model_mean_scores
                }
                df = pd.DataFrame(table_data, index=llm_ls).transpose()
                selection_order = [m for m in candidate_order if m in llm_ls] or llm_ls
                max_score_col = max(selection_order, key=lambda m: df.loc['model_mean_score', m])
                df['max_score'] = df[max_score_col]



                output_file = os.path.join(output_dir, f'{name[j]}_section{i + 1}_table.xlsx')
                df.to_excel(output_file, index=True)
                processed_tables.append(output_file)

    all_means = []
    for table_path in processed_tables:
        df = pd.read_excel(table_path, index_col=0)
        last_column_means = df.iloc[:, -1]
        all_means.append(last_column_means)

    df_means = pd.DataFrame(all_means)
    final_mean = df_means.mean()
    final_var = df_means.var()
    print(f'{baseline}: {len(processed_tables)}/{len(name) * sections} groups summarized; '
          f'{len(missing_groups)} without logs, {len(incomplete_groups)} incomplete.')
    if processed_tables:
        print('Mean over summarized groups:')
        print(final_mean.to_string())
        print('Sample variance over summarized groups:')
        print(final_var.to_string())
    else:
        print('No usable model logs; no summary statistics produced.')
