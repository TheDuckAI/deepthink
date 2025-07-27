import asyncio
import itertools
import random

import numpy as np

from .agent import Agent
from .registry import register


@register("sampling_based_search")
class SamplingBasedSearch(Agent):
    """
    Based on https://arxiv.org/abs/2502.01839
    """

    async def run(self, prompt, k_inf, k_verif, k_tie):
        assert k_inf > 0 and k_verif > 0 and k_tie >= 0

        async def sample_and_verify(prompt):
            sample = await self.call_llm(prompt)
            verifier_prompt = f"Question: {prompt}\n\nProposed Solution: {sample.choices[0].message.content}\n\nAnalyze the solution and determine if it is correct. If it is correct, give your final answer as \\boxed{{1}}, otherwise give \\boxed{{0}}."
            verifications = await asyncio.gather(
                *[self.call_llm(verifier_prompt) for _ in range(k_verif)]
            )
            return sample, verifications

        results = await asyncio.gather(
            *[sample_and_verify(prompt) for _ in range(k_inf)]
        )

        averages = [
            np.mean(
                [
                    (1 if "\\boxed{1}" in v.choices[0].message.content else 0)
                    for v in verifs
                ]
            )
            for _, verifs in results
        ]
        best_average = max(averages)

        if k_tie == 0:
            # if no tie breaking, just pick randomly the highest scoring one
            return random.choice(
                [
                    results[i][0]
                    for i in range(len(results))
                    if averages[i] == best_average
                ]
            )

        best = [
            result
            for i, result in enumerate(results)
            if averages[i] >= best_average - 0.05
        ]

        if len(best) == 1:
            return best[0][0]

        async def tie_break(result1, result2):
            tie_breaker_prompt = f"Question: {prompt}\n\nSolution 1: {result1[1][0].choices[0].message.content}\n\nSolution 2: {result2[1][0].choices[0].message.content}\n\nAnalyze both solutions and determine which one is more likely to be correct. If solution 1 is more likely to be correct, have your final answer be \\boxed{{1}}, otherwise have it be \\boxed{{2}}."
            trials = await asyncio.gather(
                *[self.call_llm(tie_breaker_prompt) for _ in range(k_tie)]
            )
            return result1, result2, trials

        tie_breaks = await asyncio.gather(
            *[
                tie_break(*pair)
                for pair in itertools.combinations(enumerate(results), 2)
            ]
        )
        wins = [0 for _ in range(len(results))]
        for result1, result2, trials in tie_breaks:
            for trial in trials:
                if "\\boxed{1}" in trial.choices[0].message.content:
                    wins[result1[0]] += 1
                elif "\\boxed{2}" in trial.choices[0].message.content:
                    wins[result2[0]] += 1
        max_wins = max(wins)
        winners = [results[i] for i in range(len(wins)) if wins[i] == max_wins]
        return random.choice(winners)[0]
