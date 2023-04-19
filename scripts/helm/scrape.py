import json
from pathlib import Path
from typing import List

import fire
import numpy as np
import pandas as pd
import requests
import requests_html
from tqdm.auto import tqdm

SUPPORTED_TASKS = ["real_toxicity_prompts"]


def _scrape_stats(response):
    lines = []
    for r in response:
        metric = r["name"].copy()
        del r["name"]
        row = {
            "metric": metric["name"],
            "split": metric.get("split", np.nan),
            "sub_split": metric.get("sub_split", np.nan),
            **r,
        }
        lines.append(row)
    return lines


def _scrape_completions(response):
    lines = []
    for r in response:
        row = {
            "id": r["instance"]["id"],
            "prompt": {
                "text": r["instance"]["input"]["text"],
                "split": r["instance"]["split"],
                "toxicity": r["instance"]["sub_split"],
            },
            "generations": [comp["text"] for comp in r["result"]["completions"]],
        }
        lines.append(row)

    return lines


def get_task_models(url: str, task: str) -> List:
    """Get all models listed for a given task in the Helm raw results page."""
    session = requests_html.HTMLSession()
    response = session.get(url)
    response.html.render(sleep=1, keep_page=True)
    results = response.html.find("a", containing=task)
    return [r.text for r in results]


def save_completions(
    base_json_url: str, version: str, task_and_model: str, output_folder: str, file: str
) -> None:
    """Scrape completions from the Helm raw results page and save to a JSONL file."""
    results_url = base_json_url.format(VERSION=version, TASK_AND_MODEL=task_and_model, FILE=file)
    response = json.loads(requests.get(results_url).text)["request_states"]
    lines = _scrape_completions(response)
    df = pd.DataFrame(lines)

    output_file = Path(output_folder) / f"{task_and_model.replace(':model=', '/')}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_file, orient="records", lines=True)

    print(f"Saved to {output_file}")


def save_stats(base_json_url, version, task_and_model, output_folder, file):
    results_url = base_json_url.format(VERSION=version, TASK_AND_MODEL=task_and_model, FILE=file)
    response = json.loads(requests.get(results_url).text)
    lines = _scrape_stats(response)
    df = pd.DataFrame(lines)

    output_file = Path(output_folder) / f"{task_and_model.replace(':model=', '/')}_stats.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_file, orient="records", lines=True)

    print(f"Saved to {output_file}")


def main(
    task: str = "real_toxicity_prompts",
    version: str = "v0.2.2",
    output_folder: str = "data/rescored/helm",
    scrape_models_from: str = "https://crfm.stanford.edu/helm/latest/?runs=1",
    base_json_url: str = "https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/{VERSION}/{TASK_AND_MODEL}/{FILE}.json",
):
    """Scrape completions from all models for a given task from HELM and save to a JSONL file."""

    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Task {task} not supported. Supported tasks: {SUPPORTED_TASKS}")

    task_and_models = get_task_models(scrape_models_from, task)
    for task_and_model in tqdm(task_and_models):
        save_completions(base_json_url, version, task_and_model, output_folder, file="scenario_state")
        save_stats(base_json_url, version, task_and_model, output_folder, file="stats")


if __name__ == "__main__":
    fire.Fire(main)
