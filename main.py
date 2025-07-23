import json
import os
import random
import re
from itertools import combinations
from typing import Callable

from datasets import load_dataset
from joblib import Parallel, delayed
from litellm import ModelResponse, completion
from math_verify import parse, verify
from tqdm_joblib import tqdm_joblib


def llm(
    model: str, messages: list[dict[str, str]], num_retries: int = 5, **sampling_params
) -> ModelResponse:
    return completion(
        model=model, messages=messages, num_retries=num_retries, **sampling_params
    )


def zeroshot(
    model: str, messages: list[dict[str, str]], **sampling_params
) -> tuple[list[ModelResponse], str]:
    completion = llm(model, messages, **sampling_params)
    return [completion], completion.choices[0].message.content


def verification(
    model: str, messages: list[dict[str, str]], k_inf, k_verif, k_tie, **sampling_params
) -> tuple[list[ModelResponse], str]:
    S = llm(model, messages, n=k_inf, **sampling_params)

    V = Parallel(n_jobs=-1)(
        delayed(llm)(
            model,
            messages=[
                {
                    "role": "user",
                    "content": f"Question: {messages[-1]['content']}\n\nProposed Solution: {s_i.message.content}\n\nIs this solution correct? If yes, respond with a final answer \\boxed{{1}}, otherwise \\boxed{{0}} if it is incorrect.",
                }
            ],
            n=k_verif,
            **sampling_params,
        )
        for s_i in S.choices
    )

    avg_scores = []
    for i, v_i in enumerate(V):
        scores = []
        for choice in v_i.choices:
            content = choice.message.content
            match = re.search(r"\\boxed\{([01])\}", content)
            if match:
                scores.append(int(match.group(1)))
            else:
                scores.append(0)
        avg_scores.append(sum(scores) / len(scores) if scores else 0)

    max_avg_score = max(avg_scores) if avg_scores else 0

    S_best_indices = []
    for i, avg_score in enumerate(avg_scores):
        if avg_score >= max_avg_score - 0.05:
            S_best_indices.append(i)

    if len(S_best_indices) == 1:
        best_idx = S_best_indices[0]
        completions = [S] + V
        return completions, S.choices[best_idx].message.content

    comparison_pairs = list(combinations(S_best_indices, 2))

    def compare_pair(pair):
        i, j = pair
        s_i = S.choices[i].message.content
        s_j = S.choices[j].message.content

        comparison_prompt = [
            {
                "role": "user",
                "content": f"Question: {messages[-1]['content']}\n\nResponse 1: {s_i}\n\nResponse 2: {s_j}\n\nWhich of these responses is more correct? If Response 1 is better, respond with \\boxed{{1}}. If Response 2 is better, respond with \\boxed{{0}}.",
            }
        ]

        C_ij = llm(model, comparison_prompt, n=k_tie, **sampling_params)
        return (i, j, C_ij)

    C_results = Parallel(n_jobs=-1)(
        delayed(compare_pair)(pair) for pair in comparison_pairs
    )

    win_counts = {i: 0 for i in S_best_indices}
    C_ijs = []

    for i, j, C_ij in C_results:
        C_ijs.append(C_ij)

        i_wins = 0
        j_wins = 0
        for choice in C_ij.choices:
            content = choice.message.content
            match = re.search(r"\\boxed\{([01])\}", content)
            if match:
                score = int(match.group(1))
                if score == 1:
                    i_wins += 1
                else:
                    j_wins += 1

        if i_wins > j_wins:
            win_counts[i] += 1
        elif j_wins > i_wins:
            win_counts[j] += 1

    winner_idx = max(win_counts, key=win_counts.get)
    completions = [S] + V + C_ijs
    return completions, S.choices[winner_idx].message.content


def eval_aime_2025(
    model: str,
    infer_fn: Callable[
        [str, list[dict[dict, str]], dict], tuple[list[ModelResponse], str]
    ],
    **sampling_params,
):
    def evaluate(problem):
        prompt = [
            {
                "role": "user",
                "content": f"Please answer the following question. At the end of your solution, put your final result in a boxed environment, e.g. \\boxed{42}.\n{problem['problem']}",
            }
        ]
        completions, result = infer_fn(model, messages=prompt, **sampling_params)

        gold = parse(problem["solution"])
        answer = parse(result)

        return {
            "model": model,
            "infer_fn": infer_fn.__name__,
            "task": "aime2025",
            "prompt": prompt,
            **sampling_params,
            "completions": [com.model_dump() for com in completions],
            "solution": result,
            "score": 1 if verify(gold, answer) else 0,
        }

    ds = load_dataset("yentinglin/aime_2025", split="train")
    with tqdm_joblib(
        desc="Evaluation progress",
        total=len(ds),
        unit="problem",
        dynamic_ncols=True,
        colour=f"#{random.randint(0, 16777215):06x}",
    ) as _:
        results = Parallel(n_jobs=os.cpu_count())(
            delayed(evaluate)(problem) for problem in ds
        )

    return results


def main():
    model = "gemini/gemini-2.5-flash-lite"
    infer_fn = verification
    sampling_params = dict(k_inf=5, k_verif=3, k_tie=2)
    eval_results = eval_aime_2025(model, infer_fn, **sampling_params)
    print(eval_results)
    with open(
        f"eval_results_aime2025_{model.replace('/', '__')}_{infer_fn.__name__}.jsonl",
        "w",
    ) as f:
        for result in eval_results:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
