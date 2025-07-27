import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
from joblib import Parallel, delayed
from math_verify import parse, verify
from tqdm_joblib import tqdm_joblib


def eval_aime(solution, ground_truth):
    gold = parse(ground_truth)
    solution = parse(solution)
    return 1 if verify(gold, solution) else 0


def parse_args():
    parser = ArgumentParser(
        prog=f"uv run {os.path.basename(__file__)}",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.input) as f:
        gens = [json.loads(line) for line in f.readlines()]

    if len(gens) == 0:
        raise ValueError("no generations to evaluate")

    print(f"Loaded {len(gens)} generations")

    tasks = []
    for gen in gens:
        match gen["dataset"]:
            case "aime2025":
                tasks.append(
                    delayed(eval_aime)(
                        gen["final_response"]["choices"][0]["message"]["content"],
                        gen["answer"],
                    )
                )
            case _:
                raise NotImplementedError()

    with tqdm_joblib(
        desc="Evaluation progress",
        total=len(tasks),
        unit="eval",
        dynamic_ncols=True,
    ) as _:
        results = Parallel(n_jobs=-1)(tasks)

    results = np.array(results)
    mean_acc = results.mean()
    # bootstrap
    n_boot = 10_000
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = np.random.choice(results, size=results.size, replace=True)
        boot_means[i] = sample.mean()

    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    print(f"Point estimate: {mean_acc:.4f}")
    print(f"95% CI        : [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
