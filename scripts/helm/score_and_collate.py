from pathlib import Path

import fire
from tqdm.auto import tqdm

from scripts.collate import main as collate
from scripts.score import main as score


def main(perspective_rate_limit, base_dir: str = "data/rescored/helm/real_toxicity_prompts", prompts_path: str = None) -> None:
    """Score and collate all JSONL files in a given directory."""
    base_dir = Path(base_dir)
    files = [
        f
        for f in base_dir.glob("*.jsonl")
        if all(s not in f.name for s in ["perspective", "collated", "stats"])
    ]

    for file in tqdm(files):
        scores_path = score(input_filename=file, perspective_rate_limit=perspective_rate_limit)
        prompts_path = prompts_path or file
        collate(
            generations_path=file,
            scores_path=scores_path,
            prompts_path=prompts_path,
            results_from_span=True
        )


if __name__ == "__main__":
    fire.Fire(main)
