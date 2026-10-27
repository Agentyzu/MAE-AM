import argparse
import copy
import json
import random
from importlib import import_module
from pathlib import Path

import numpy as np

from model_zoo.model_zoo import ModelZoo
from utils.api_config import require_env


ROOT = Path(__file__).resolve().parent
AUCTIONS = {"Prom": "Prom", "GSP": "GSP", "GA": "GA", "SQA": "GSP_SQA",
            "ALS": "Prom_ALS", "AAL": "Prom_AAL", "Prom_DSIC": "Prom_DSIC"}


def main(argv=None, dataset="ADGEN", auction="Prom"):
    parser = argparse.ArgumentParser(description="Run auction-guided advertising experiments")
    parser.add_argument("--dataset", choices=("ADGEN", "ATVI"), default=dataset)
    parser.add_argument("--auction", choices=tuple(AUCTIONS), default=auction)
    parser.add_argument("--data-dir", type=Path, help="Dataset directory containing category subdirectories")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--categories", nargs="+", help="Category names; default: all categories with configs")
    parser.add_argument("--sections", nargs="+", type=int, help="Split numbers; default: all available splits")
    parser.add_argument("--models", nargs="+", choices=tuple(ModelZoo.available_llms))
    parser.add_argument("--language", choices=("chinese", "english"))
    parser.add_argument("--content-field", help="Override the advertisement content field")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.sections and any(section < 1 for section in args.sections):
        parser.error("Split numbers must be positive")

    data_dir = (args.data_dir or ROOT / "dataset" / args.dataset).resolve()
    config_auction = "Prom" if args.auction == "Prom_DSIC" else args.auction
    config_dir = ROOT / "cfgs" / f"{args.dataset}_{config_auction}_configs"
    if not data_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {data_dir}")
    configured = {path.stem[len('config_'):-len(f'_{config_auction}')]
                  for path in config_dir.glob(f"config_*_{config_auction}.py")}
    categories = sorted(args.categories or (path.name for path in data_dir.iterdir()
                                             if path.is_dir() and path.name in configured))
    jobs = []
    for category in categories:
        if category not in configured:
            parser.error(f"No {args.dataset}/{args.auction} config for category: {category}")
        folder = data_dir / category
        if not folder.is_dir():
            parser.error(f"Category directory does not exist: {folder}")
        if args.sections:
            files = [folder / f"{category}_section{section}.json" for section in sorted(set(args.sections))]
        else:
            files = sorted(folder.glob(f"{category}_section*.json"),
                           key=lambda path: int(path.stem.rsplit('section', 1)[1]))
        for path in files:
            if path.is_file():
                jobs.append((category, path))
            else:
                print(f"Missing split, skipped: {path}")
    if not jobs:
        parser.error("No experiment splits found for the selected categories and sections")

    require_env('SIMILARITY_URL', 'SIMILARITY_AUTH_URL', 'SIMILARITY_API_KEY', 'SIMILARITY_SECRET_KEY')
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Running {len(jobs)} split(s): {args.dataset}/{args.auction}; seed={args.seed}")
    for category, path in jobs:
        module = import_module(f"cfgs.{args.dataset}_{config_auction}_configs.config_{category}_{config_auction}")
        cfg = copy.deepcopy(module.cfg)
        cfg.auc = AUCTIONS[args.auction]
        cfg.dataset = args.dataset
        cfg.language = args.language or ("chinese" if args.dataset == "ADGEN" else "english")
        cfg.output_dir = str(args.output_dir.resolve())
        cfg.seed = args.seed
        if args.models:
            cfg.models = args.models
        with path.open(encoding="utf-8-sig") as file:
            data = json.load(file)
        if not isinstance(data, list) or not data or not all(isinstance(row, dict) for row in data):
            raise ValueError(f"Expected a nonempty JSON array of advertisement objects: {path}")
        cfg.content_field = args.content_field or ("content" if "content" in data[0] else "ad_copy")
        required = {"ad_name", "pctr", cfg.content_field}
        if any(not required <= row.keys() for row in data):
            raise ValueError(f"Missing required advertisement fields {sorted(required)}: {path}")
        if any(not isinstance(row[cfg.content_field], str) for row in data):
            raise ValueError(f"Advertisement content must be text: {path}")
        ModelZoo(cfg, data, path.stem).run_all_models()


if __name__ == "__main__":
    main()
