import json
import os
import random
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
) -> list[ModelResponse]:
    return [llm(model, messages, **sampling_params)]


def deepthink(
    model: str, messages: list[dict[str, str]], k_inf, k_verif, k_tie, **sampling_params
) -> list[ModelResponse]:
    S = llm(model, messages, n=k_inf)
    V = []
    for s_i in S.choices:
        V.append(llm(model, messages, n=k_verif))


def eval_aime_2025(
    model: str,
    infer_fn: Callable[[str, list[dict[dict, str]], dict], list[ModelResponse]],
    **sampling_params,
):
    def evaluate(problem):
        prompt = [
            {
                "role": "user",
                "content": f"""Please answer the following question. Think carefully and in a step-by-step fashion. At the end of
your solution, put your final result in a boxed environment, e.g. \\boxed{42}.\n{problem["problem"]}""",
            }
        ]
        result: list[ModelResponse] = infer_fn(
            model, messages=prompt, **sampling_params
        )

        gold = parse(problem["solution"])
        answer = parse(result[-1].choices[0].message.content)

        return {
            "model": model,
            "infer_fn": infer_fn.__name__,
            "task": "aime2025",
            "prompt": prompt,
            **sampling_params,
            "completions": [res.model_dump() for res in result],
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
    eval_results = eval_aime_2025("o4-mini", zeroshot)
    print(eval_results)
    with open("eval_results_o4_mini.jsonl", "w") as f:
        for result in eval_results:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
