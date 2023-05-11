# On the Challenges of Using Black-Box APIs for Toxicity Evaluation in Research

Published on the [Trustworthy and Reliable Large-Scale Machine Learning Models ICLR 2023 Workshop](https://rtml-iclr2023.github.io/cfp.html).

[[Data]](https://huggingface.co/datasets/for-ai/black-box-api-challenges) [[OpenReview]](https://openreview.net/forum?id=bRDHL4J5vy) [[Arxiv]]()

**Abstract**: Perception of toxicity evolves over time and often differs between geographies and cultural backgrounds. Similarly, black-box commercially available APIs for detecting toxicity, such as the Perspective API, are not static, but frequently retrained to address any unattended weaknesses and biases. We evaluate the implications of these changes on the reproducibility of findings that compare the relative merits of models and methods that aim to curb toxicity. Our findings suggest that research that relied on inherited automatic toxicity scores to compare models and techniques may have resulted in inaccurate findings. Rescoring all models from HELM, a widely respected living benchmark, for toxicity with the recent version of the API led to a different ranking of widely used foundation models. We suggest caution in applying apples-to-apples comparisons between studies and lay recommendations for a more structured approach to evaluating toxicity over time.

## Installing

All images, tables and values cited in the paper can be reproduced in the notebooks 01 and 02.

```bash
conda env create -f environment.yml
conda init black_box
python -m ipykernel install --user --name=black_box
```

### Download data

Rescored toxicity scores and metrics produced for the paper are available at our HuggingFace datasets repo. Published scores from RTP are also needed to reproduce results.

```bash
git lfs install
git clone git@hf.co:datasets/for-ai/black-box-api-challenges data
```

```bash
wget https://ai2-public-datasets.s3.amazonaws.com/realtoxicityprompts/realtoxicityprompts-data.tar.gz
tar -xvzf realtoxicityprompts-data.tar.gz -C data/
rm realtoxicityprompts-data.tar.gz
```

## Run scripts

There are three main scripts: `score`, `collate` and `evaluate`. Below are examples of how to use each for DExperts rescored generation files that accompany this repo.

### Score

You can replace the `input_path` for your desired `jsonl` file and indicate in which column are the text you want to rescore. The script currently supports text that are contained in dictionaries (`text` key), list of dictionaries and columns of strings. This outputs files with `_perspective.jsonl` termination.

Perspective API rate limit is 1 by default. Before running this script, don't forget to export your API key.

```bash
export PERSPECTIVE_API_KEY=$YOUR_KEY
```

```bash
python -m scripts.score \
    data/dexperts/generations/toxicity/dapt/prompted_gens_gpt2_gens_rescored.jsonl \
    --column_name generations \
    --output_folder data/example \
    --perspective_rate_limit 1
```

To rescore DExperts's 10k non-toxic RTP prompts, for example, you can run

```bash
python -m scripts.score \
    data/dexperts/prompts/nontoxic_prompts-10k.jsonl \
    --column_name prompt \
    --output_folder data/example \
    --perspective_rate_limit 1
```

### Collate

The collate script joins prompts and generations into a single file. We need all **three files**: with generated text, with scores corresponding to those texts, and the prompts which generated the continuations, if used. This outputs files with `_collated.jsonl` termination.

```bash
python -m scripts.collate \
    data/dexperts/generations/toxicity/dapt/prompted_gens_gpt2_gens_rescored.jsonl \
    data/example/prompted_gens_gpt2_gens_rescored_perspective.jsonl \
    --prompts_path data/dexperts/prompts/nontoxic_prompts-10k.jsonl
```

You can collate prompts to their new scores with

```python
python -m scripts.collate_prompts data/dexperts/prompts/nontoxic_prompts-10k.jsonl data/example/nontoxic_prompts-10k_perspective.jsonl
```

### Evaluate

With the evaluate script we can compute **toxicity metrics** such as Expected Maximum Toxicity, Toxicity Probability and Toxic Fraction This outputs files with `_toxicity.csv` termination.

```bash
python -m scripts.evaluate --prompted_json data/example/prompted_gens_gpt2_gens_rescored_collated.jsonl
```

## Scrape, score and evaluate HELM continuations

We scrape the website for models benchmarked under the `real_toxicity_prompts` task.
Those model names are used to download continuations and published `stats.jsonl` files from HELM's buckets.

```bash
python -m scripts.helm.scrape \
    --task "real_toxicity_prompts" \
    --version "v0.2.2" \
    --output_folder "data/rescored/helm"
```

Then, we rescore downloaded continuations and collate those scores with the original prompts. You can pass `prompts_path` in case you rescored prompts as well.

HELM uses the first `spanScore`, instead of the `summaryScores` as other RTP benchmarks. _Evaluation is performed exclusively on notebook 02_ from the collated files and original `stats.jsonl` files.

```bash
python -m scripts.helm.score_and_collate \
    --perspective_rate_limit 1 \
    --base_dir data/rescored/helm/real_toxicity_prompts
```

# Citation

```
@article{pozzobon2023challenges,
  title={On the Challenges of Using Black-Box APIs for Toxicity Evaluation in Research},
  author={Pozzobon, Luiza and Ermis, Beyza and Lewis, Patrick and Hooker, Sara},
  journal={arXiv preprint arXiv:2304.12397},
  year={2023}
}
```

# Useful repos

- **RealToxicityPrompts**: https://github.com/allenai/real-toxicity-prompts
- **DExperts**: https://github.com/stanford-crfm/helm
- **HELM**: https://github.com/alisawuffles/DExperts
