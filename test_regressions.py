import ast
import io
import json
import os
import tokenize
from contextlib import redirect_stdout
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from auction.GA import GA
from auction.GSP_SQA import GSP_SQA
from auction.Prom import Prom
from auction.Prom_DSIC import Prom_DSIC
from utils.base_llm import Base_LLM
from utils.LLMProcess import LLMProcess


ROOT = Path(__file__).resolve().parent


class Config(dict):
    __getattr__ = dict.__getitem__


class OfflineLLM(Base_LLM):
    def reply(self, query):
        return "offline answer"


def inputs(auc="Prom"):
    cfg = Config(q=2, constant=3, max_len=100, query="test query",
                 is_structured=True, theta_1=0.5, theta_2=0.6,
                 models=["Qwen"], auc=auc)
    data = [{"ad_name": str(i), "content": content, "pctr": 1.0}
            for i, content in enumerate(["A" * 5, "B" * 150, "C" * 150])]
    return cfg, data


def check_similarity_and_satisfaction():
    cfg, data = inputs()
    data[0]["content"] = " ".join(["word"] * 60)
    llm = OfflineLLM(cfg, data, Prom)
    original = llm.content.copy()
    with patch("utils.base_llm.sim", side_effect=[0.2, 0.9, 0.5]) as similarity:
        llm.calculate_similar(" ".join(["answer"] * 60))
        assert all(len(arg.split()) == 50 for arg in similarity.call_args_list[0].args)
    np.testing.assert_allclose(llm.b, [0.2, 0.9, 0.5])
    assert llm.content == original
    llm.auction()
    np.testing.assert_array_equal(llm.sigma[:2], [1, 2])
    llm.basic_reply = "base answer"
    llm.compressed_ad = "final answer"
    llm.ad_len = [0, 5]
    llm.SW = 0
    with patch("utils.base_llm.sim", return_value=0.8) as similarity:
        llm.satisfaction()
        similarity.assert_called_once_with("base answer", "final answer")
    assert llm.ad_satis == 0 and llm.user_satis == 0.8 and llm.score == 0.4


def check_dsic():
    bids = np.array([0.4, 0.9, 0.7])
    pctr = np.ones(3)
    positions = np.array([0.9, 0.81])
    args = (pctr, positions, 2, 3, [100] * 3, 100)
    truthful = Prom(bids, *args)
    for s in range(3):
        reported = Prom_DSIC(bids, *args, 1, s)
        for actual, expected in zip(reported, truthful):
            np.testing.assert_allclose(actual, expected)
    result = Prom_DSIC(bids, *args, 3, 0)
    np.testing.assert_array_equal(result[0][:2], [0, 1])
    np.testing.assert_allclose(result[3], bids[result[0][:2]] * result[1] - result[2])
    true_values = bids[result[0][:2]] * positions
    np.testing.assert_allclose(result[4], np.sum(true_values * (np.log(result[1]) + 3)))
    np.testing.assert_array_equal(Prom_DSIC(bids, *args, 0.1, 1)[0][:2], [2, 0])
    np.testing.assert_array_equal(bids, [0.4, 0.9, 0.7])
    q1 = Prom_DSIC(bids, pctr, [1], 1, 3, [100] * 3, 100, 3, 0)
    assert q1[0][0] == 0 and q1[2][0] == 0.9
    cfg, data = inputs("Prom_DSIC")
    process = LLMProcess(OfflineLLM, cfg, data, Prom_DSIC)
    with patch("utils.base_llm.sim", return_value=0.5):
        process.run_DSIC()
    assert len([key for key in process.result if key.startswith("ad_rank_")]) == 27
    assert "ad_rank_1_2" in process.result


def check_allocations_and_generation():
    bids = np.array([0.9, 0.8, 0.7])
    for auction in (GA, GSP_SQA):
        for lengths, budget in (([200, 200, 200], 100), ([0, 95, 100], 100),
                                ([5, 95, 100], 100), ([1, 99, 100], 100),
                                ([0, 0, 0], 100), ([100] * 3, 0)):
            rank, prom, payments, utilities, welfare = auction(
                bids, np.ones(3), np.ones(3), 3, 3, lengths, budget)
            assert prom.shape == payments.shape == utilities.shape == (3,)
            assert np.sum(prom) <= 1 + 1e-12 and np.all(prom >= 0)
            assert sum(int(budget * p) for p in prom) <= budget
            assert np.all(payments[prom == 0] == 0) and np.all(utilities[prom == 0] == 0)
            assert all(np.all(np.isfinite(x)) for x in (prom, payments, utilities, welfare))
            np.testing.assert_allclose(utilities[prom > 0],
                                       bids[rank[:3]][prom > 0] * prom[prom > 0] - payments[prom > 0])
        single = auction(np.array([1.0]), [1.0], [1.0], 1, 3, [100], 100)
        assert single[2][0] == 0
    cfg, data = inputs()
    llm = OfflineLLM(cfg, data, GA)
    llm.basic_reply = "base"
    llm.sigma = np.array([0, 1, 2])
    for method in ("ad_gen_chinese_GA", "ad_gen_english_GA",
                   "ad_gen_chinese_SQA", "ad_gen_english_SQA"):
        for lengths in ([0, 95], [5, 95]):
            llm.ad_len = lengths
            output = getattr(llm, method)()
            assert output.count("A") == lengths[0] and output.count("B") == 95
            assert "C" not in output
        llm.ad_len = [0, 0]
        assert getattr(llm, method)() == ("base" if method.endswith("SQA") else "")


def check_zoo_flows():
    cases = [("model_zoo.py", "ModelZoo", auc, flow) for auc, flow in
             [("Prom", "run"), ("GSP", "run"), ("GA", "run_ga"),
              ("GSP_SQA", "run_sqa"), ("Prom_DSIC", "run_DSIC")]]
    cases += [(f"model_zoo_{suffix}.py", f"ModelZoo_{suffix}", auc, flow)
              for suffix, auc, flow in [("AAL", "Prom_AAL", "run_aal"),
                                        ("ALS", "Prom_ALS", "run_als"),
                                        ("GA", "GA", "run_ga"),
                                        ("SQA", "GSP_SQA", "run_sqa"),
                                        ("DSIC", "Prom_DSIC", "run_DSIC")]]
    for filename, class_name, auc, flow in cases:
        path = ROOT / "model_zoo" / filename
        module = import_module(f"model_zoo.{path.stem}")
        cfg, data = inputs(auc)
        with patch("model_zoo.model_zoo.import_module", return_value=SimpleNamespace(Qwen=OfflineLLM)):
            zoo = getattr(module, class_name)(cfg, data, "offline")
        zoo.package_results = Mock()
        process = zoo.llms["Qwen"]
        with patch("utils.base_llm.sim", return_value=0.5), \
                patch("numpy.random.uniform", return_value=np.array([0.5] * 3)), \
                patch.object(process, flow, wraps=getattr(process, flow)) as run:
            single = zoo.run_model("Qwen").copy()
            zoo.run_all_models()
        assert run.call_count == 2
        assert single == process.result
        assert zoo.package_results.call_count == 2
        zoo.package_results.assert_called_with(process.result, "Qwen")
        json.dumps(process.result)


def check_zero_weight_auctions():
    cases = [(np.zeros(3), np.ones(3), np.ones(3)),
             (np.array([0.8, 0.4, 0.0]), np.ones(3), np.ones(3)),
             (np.array([1.0, 0.0, 0.0]), np.ones(3), np.ones(3)),
             (np.ones(3), np.zeros(3), np.ones(3)),
             (np.ones(3), np.ones(3), np.array([1.0, 0.0, 0.0]))]
    for bids, pctr, positions in cases:
        truthful = Prom(bids, pctr, positions, 3, 3, [100] * 3, 100)
        for auction in (Prom, Prom_DSIC):
            extra = (1, 0) if auction is Prom_DSIC else ()
            rank, prom, payments, utilities, welfare = auction(
                bids, pctr, positions, 3, 3, [100] * 3, 100, *extra)
            assert all(np.all(np.isfinite(value)) for value in (prom, payments, utilities, welfare))
            assert np.all(payments[prom == 0] == 0) and np.all(utilities[prom == 0] == 0)
            if np.count_nonzero(prom) <= 1:
                assert np.all(payments == 0)
            values = bids[rank[:3]] * pctr[rank[:3]] * positions
            allocated = prom > 0
            np.testing.assert_allclose(welfare, np.sum(values[allocated] * (np.log(prom[allocated]) + 3)))
            if not np.any(values):
                assert np.all(prom == 0) and welfare == 0
            else:
                np.testing.assert_allclose(prom.sum(), 1)
            for actual, expected in zip((rank, prom, payments, utilities, welfare), truthful):
                np.testing.assert_allclose(actual, expected)
    for auction in (Prom, Prom_DSIC):
        extra = (1, 0) if auction is Prom_DSIC else ()
        for bid in (0.0, 1.0):
            result = auction(np.array([bid]), [1.0], [1.0], 1, 3, [100], 100, *extra)
            assert result[1][0] == bid and result[2][0] == 0 and result[3][0] == bid
            assert result[4] == 3 * bid
        extra = (0, 0) if auction is Prom_DSIC else ()
        bids = np.array([1.0]) if extra else np.array([0.0])
        result = auction(bids, [1.0], [1.0], 1, 3, [100], 100, *extra)
        assert all(np.all(value == 0) for value in result[1:])
    for auction, method in ((Prom, 'run'), (Prom_DSIC, 'run_DSIC')):
        cfg, data = inputs()
        cfg['q'] = 3
        process = LLMProcess(OfflineLLM, cfg, data, auction)
        with patch('utils.base_llm.sim', return_value=0.0):
            getattr(process, method)()
        assert process.llm.ad_len == [0, 0, 0]
        json.dumps(process.result, allow_nan=False)


def check_result_serialization():
    from model_zoo.model_zoo import ModelZoo

    with TemporaryDirectory(dir=ROOT) as directory:
        zoo = ModelZoo.__new__(ModelZoo)
        zoo.cfg = {'output_dir': directory, 'dataset': 'ADGEN', 'auc': 'Prom'}
        zoo.data_name = 'synthetic'
        result = {'ad_sw': 1.0, 'reply': 'valid result'}
        zoo.package_results(result, 'offline')
        path = Path(directory) / 'ADGEN' / 'Prom' / 'synthetic' / 'offline_results_synthetic.json'
        original = path.read_bytes()
        for bad in (float('nan'), float('inf'), object()):
            for model in ('offline', 'new'):
                try:
                    zoo.package_results({'ad_sw': bad}, model)
                except (TypeError, ValueError):
                    pass
                else:
                    raise AssertionError('Invalid result was serialized')
                assert path.read_bytes() == original
                assert not (path.parent / 'new_results_synthetic.json').exists()
        assert json.loads(path.read_text(encoding='utf-8')) == result


def check_integer_budget_recovery():
    for auction in (GA, GSP_SQA):
        for length in range(1, 100):
            cfg, data = inputs()
            data[0]['content'] = 'A' * length
            model = OfflineLLM(cfg, data, auction)
            model.b = np.array([0.9, 0.8, 0.7])
            model.auction()
            expected = [length, 100 - length] if auction is GA else [length, 0]
            assert model.ad_len == expected
            assert sum(model.ad_len) <= model.max_len
    cfg, data = inputs()
    model = OfflineLLM(cfg, data, Prom)
    model.b = np.array([0.9, 0.8, 0.7])
    model.auction()
    assert model.ad_len == [int(100 * prominence) for prominence in model.prom]
    assert sum(model.ad_len) <= model.max_len


def check_dsic_utility_summary():
    from utils.data_process import collect_results, main as aggregate
    import pandas as pd

    bids = np.array([0.4, 0.9, 0.7])
    result = {}
    expected = {}
    for advertiser in (0, 1):
        for beta in (2.5, 3, 0.1, 1):
            rank, prom, payments, utilities, welfare = Prom_DSIC(
                bids, np.ones(3), np.array([0.9, 0.81]), 2, 3, [100] * 3, 100, beta, advertiser)
            suffix = f'_{beta}_{advertiser}'
            result[f'ad_rank{suffix}'] = rank.tolist()
            result[f'ad_utility{suffix}'] = utilities.tolist()
            result[f'ad_rev{suffix}'] = payments.tolist()
            result[f'ad_sw{suffix}'] = float(welfare)
            allocated = rank[:len(utilities)].tolist()
            expected[beta, advertiser] = float(utilities[allocated.index(advertiser)]) if advertiser in allocated else 0.0
    with TemporaryDirectory(dir=ROOT) as directory:
        source = Path(directory) / 'ADGEN' / 'Prom_DSIC' / 'synthetic_section1'
        source.mkdir(parents=True)
        path = source / 'offline_results_synthetic_section1.json'
        path.write_text(json.dumps(result), encoding='utf-8')
        for spelling in ('1', '1.0'):
            serialized = {key.replace('_1_', f'_{spelling}_'): value for key, value in result.items()}
            path.write_text(json.dumps(serialized), encoding='utf-8')
            table = collect_results(source.parent)
            for row in table.to_dict('records'):
                advertiser = row['advertiser']
                baseline = expected[1, advertiser]
                assert row['advertiser_utility'] == expected[row['beta'], advertiser]
                assert row['truthful_utility'] == baseline
                np.testing.assert_allclose(row['utility_gain'], row['advertiser_utility'] - baseline)
            assert table.loc[(table.beta == 1) & (table.advertiser == 0), 'advertiser_utility'].iloc[0] == 0
            assert table.loc[(table.beta == 2.5) & (table.advertiser == 0), 'utility_gain'].iloc[0] > 0
            assert table.loc[(table.beta == 3) & (table.advertiser == 0), 'utility_gain'].iloc[0] < 0
            assert table.loc[(table.beta == 0.1) & (table.advertiser == 1), 'utility_gain'].iloc[0] < 0
        aggregate(['--results-dir', directory, '--dataset', 'ADGEN', '--auction', 'Prom_DSIC'])
        sheets = pd.read_excel(source.parent / 'summary.xlsx', sheet_name=None)
        columns = {'advertiser_utility', 'truthful_utility', 'utility_gain'}
        assert all(columns <= set(table.columns) for table in sheets.values())
        missing_baseline = {key: value for key, value in result.items() if '_1_' not in key}
        path.write_text(json.dumps(missing_baseline), encoding='utf-8')
        try:
            collect_results(source.parent)
        except ValueError as error:
            assert 'beta=1 baseline' in str(error)
        else:
            raise AssertionError('Missing truthful baseline was accepted')


SIMILARITY_ENV = {
    'SIMILARITY_URL': 'https://example.invalid/similarity',
    'SIMILARITY_AUTH_URL': 'https://example.invalid/auth',
    'SIMILARITY_API_KEY': 'offline-placeholder',
    'SIMILARITY_SECRET_KEY': 'offline-placeholder',
    'API_TIMEOUT_SECONDS': '12'
}


def check_api_configuration():
    from utils.api_config import require_env, request_timeout
    from utils.calculate_sim import sim

    with patch.dict(os.environ, {}, clear=True), patch('requests.post') as post:
        try:
            sim('a', 'b')
        except ValueError as error:
            assert 'SIMILARITY_URL' in str(error)
        else:
            raise AssertionError('Missing environment variables were accepted')
        post.assert_not_called()
        assert request_timeout() == 60
    for value in ('0', '-1', 'nan', 'inf'):
        with patch.dict(os.environ, {'API_TIMEOUT_SECONDS': value}):
            try:
                request_timeout()
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid timeout was accepted')
    auth = Mock()
    auth.json.return_value = {'access_token': 'offline-token'}
    response = Mock()
    response.json.return_value = {'score': 0.75}
    with patch.dict(os.environ, SIMILARITY_ENV, clear=True), \
            patch('requests.post', side_effect=[auth, response]) as post:
        assert sim('a', 'b') == 0.75
        assert post.call_count == 2
        assert all(call.kwargs['timeout'] == 12 for call in post.call_args_list)
        assert post.call_args.kwargs['params'] == {'access_token': 'offline-token'}
        auth.raise_for_status.assert_called_once()
        response.raise_for_status.assert_called_once()
    response.json.return_value = {}
    with patch.dict(os.environ, SIMILARITY_ENV, clear=True), \
            patch('requests.post', side_effect=[auth, response]):
        try:
            sim('a', 'b')
        except ValueError as error:
            assert 'score' in str(error)
        else:
            raise AssertionError('Missing score was accepted')
    from LLMs.Qwen import Qwen

    cfg, data = inputs()
    with patch.dict(os.environ, {'QWEN_API_KEY': 'offline-placeholder',
                                'QWEN_BASE_URL': 'https://example.invalid/v1',
                                'QWEN_MODEL': 'offline-model'}, clear=True), \
            patch('LLMs.Qwen.OpenAI') as client:
        model = Qwen(cfg, data, Prom)
        client.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='offline reply'))])
        assert model.reply('query') == 'offline reply'
        assert client.call_count == 1
        assert client.call_args.kwargs['max_retries'] == 0
        assert client.call_args.kwargs['timeout'] == 60
        assert client.return_value.chat.completions.create.call_args.kwargs['model'] == 'offline-model'
        assert model.messages[-1]['role'] == 'assistant'
    with patch.dict(os.environ, {}, clear=True):
        try:
            require_env('QWEN_API_KEY')
        except ValueError:
            pass
        else:
            raise AssertionError('Missing API key was accepted')


def check_runner_and_export():
    import run_experiment
    from utils.data_process import collect_results, main as aggregate
    import pandas as pd

    class Settings(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as error:
                raise AttributeError(name) from error

        def __setattr__(self, name, value):
            self[name] = value

    with TemporaryDirectory(dir=ROOT) as directory:
        folder = Path(directory)
        output = folder / 'results'
        adapters = dict(run_experiment.ModelZoo.available_llms)
        cfg, data = inputs()
        cfg = Settings(cfg)
        for dataset, category in [('ADGEN', 'jacket'), ('ATVI', 'Travel')]:
            category_dir = folder / dataset / category
            category_dir.mkdir(parents=True)
            rows = [dict(row) for row in data]
            if dataset == 'ATVI':
                for row in rows:
                    row['ad_copy'] = row.pop('content')
            for section in (2, 1):
                (category_dir / f'{category}_section{section}.json').write_text(json.dumps(rows), encoding='utf-8')
        original_cwd = Path.cwd()
        os.chdir(folder)
        try:
            with patch.dict(os.environ, SIMILARITY_ENV), \
                    patch('run_experiment.import_module', return_value=SimpleNamespace(cfg=cfg)), \
                    patch('model_zoo.model_zoo.import_module', return_value=SimpleNamespace(Qwen=OfflineLLM)), \
                    patch('utils.base_llm.sim', return_value=0.5):
                for dataset in ('ADGEN', 'ATVI'):
                    for auction in ('Prom', 'Prom_DSIC'):
                        run_experiment.main(['--dataset', dataset, '--auction', auction,
                                             '--data-dir', str(folder / dataset),
                                             '--output-dir', str(output)])
            for dataset, category, field, language in [('ADGEN', 'jacket', 'content', 'chinese'),
                                                        ('ATVI', 'Travel', 'ad_copy', 'english')]:
                for auction in ('Prom', 'Prom_DSIC'):
                    source = output / dataset / auction
                    files = list(source.rglob('*.json'))
                    assert len(files) == 2
                    result = json.loads(files[0].read_text(encoding='utf-8'))
                    assert result['config']['language'] == language
                    assert result['config']['content_field'] == field
                    assert result['config']['auc'] == auction
                    assert result['bids'] == [0.5] * 3
                    table = collect_results(source)
                    assert len(table) == (2 if auction == 'Prom' else 54)
                    aggregate(['--dataset', dataset, '--auction', auction, '--results-dir', str(output)])
                    sheets = pd.read_excel(source / 'summary.xlsx', sheet_name=None)
                    assert set(sheets) == {'per_split', 'mean', 'variance', 'count'}
            for alias, name in [('ALS', 'Prom_ALS'), ('AAL', 'Prom_AAL'), ('SQA', 'GSP_SQA')]:
                source = output / 'ADGEN' / name / 'jacket_section1'
                source.mkdir(parents=True)
                original = output / 'ADGEN' / 'Prom' / 'jacket_section1' / 'Qwen_results_jacket_section1.json'
                (source / original.name).write_text(original.read_text(encoding='utf-8'), encoding='utf-8')
                aggregate(['--dataset', 'ADGEN', '--auction', alias, '--results-dir', str(output)])
                assert (source.parent / 'summary.xlsx').is_file()
            with patch('run_experiment.ModelZoo', wraps=run_experiment.ModelZoo) as zoo, \
                    patch('run_experiment.import_module', return_value=SimpleNamespace(cfg=cfg)), \
                    patch('model_zoo.model_zoo.import_module', return_value=SimpleNamespace(Qwen=OfflineLLM)), \
                    patch('utils.base_llm.sim', return_value=0.5), patch.dict(os.environ, SIMILARITY_ENV):
                zoo.available_llms = adapters
                run_experiment.main(['--data-dir', str(folder / 'ADGEN'), '--output-dir', str(output),
                                     '--categories', 'jacket', '--sections', '2', '--models', 'Qwen'])
                assert zoo.call_count == 1 and zoo.call_args.args[2] == 'jacket_section2'
            with patch('run_experiment.main') as entry:
                for package in ('run_ADGEN', 'run_ATVI'):
                    for path in (ROOT / package).glob('*.py'):
                        import_module(f'{package}.{path.stem}')
                entry.assert_not_called()
        finally:
            os.chdir(original_cwd)


def check_english_comments():
    paths = list(ROOT.glob('*.py'))
    for directory in ('LLMs', 'auction', 'model_zoo', 'utils', 'cfgs', 'run_ADGEN', 'run_ATVI'):
        paths.extend((ROOT / directory).rglob('*.py'))
    for path in paths:
        text = path.read_text(encoding='utf-8-sig')
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                assert not any('\u4e00' <= char <= '\u9fff' for char in token.string), f'Chinese comment: {path}:{token.start[0]}'
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node) or ''
                assert not any('\u4e00' <= char <= '\u9fff' for char in doc), f'Chinese docstring: {path}'


if __name__ == "__main__":
    with redirect_stdout(io.StringIO()), patch('utils.base_llm.tqdm', side_effect=lambda items: items), \
            np.errstate(divide="raise", invalid="raise"):
        check_similarity_and_satisfaction()
        check_dsic()
        check_allocations_and_generation()
        check_zero_weight_auctions()
        check_result_serialization()
        check_integer_budget_recovery()
        check_dsic_utility_summary()
        check_zoo_flows()
        check_api_configuration()
        check_runner_and_export()
        check_english_comments()
    print("PASS: auction regressions, API configuration, multilingual runner, JSON/Excel export, and English comments")
