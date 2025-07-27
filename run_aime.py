import asyncio
import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from datasets import load_dataset
from openai.types.chat import ChatCompletion
from tqdm import tqdm

from deepthink.agent import Agent
from deepthink.llm import LLM
from deepthink.registry import get_agent_cls, list_agents


def parse_args():
    parser = ArgumentParser(
        prog=f"uv run {os.path.basename(__file__)}",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model", type=str, help="base model to use")
    parser.add_argument(
        "strategy",
        type=str,
        help="inference strategy to use",
        choices=list_agents(),
    )
    parser.add_argument(
        "--base_url",
        type=str,
        help="openai-api-compatible api endpoint",
        default="https://openrouter.ai/api/v1",
    )
    parser.add_argument(
        "--api_key", type=str, help="api key", default=os.environ["OPENROUTER_API_KEY"]
    )
    parser.add_argument(
        "--n",
        type=int,
        help="number of samples per question",
        default=1,
    )
    parser.add_argument(
        "--extra_body",
        type=json.loads,
        help="extra body for openai request",
        default={},
    )
    parser.add_argument("--output", type=str, help="output file", required=False)
    args, extra_args = parser.parse_known_args()
    return args, parse_kwargs(extra_args)


def auto_cast(val):
    if val.lower() in {"true", "false"}:
        return val.lower() == "true"
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def parse_kwargs(extra_args):
    kwargs = {}
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                value = auto_cast(extra_args[i + 1])
                i += 2
            else:
                value = True
                i += 1
            kwargs[key] = value
        else:
            i += 1
    return kwargs


async def main():
    args, extra_args = parse_args()

    dataset = load_dataset("yentinglin/aime_2025", split="train")

    llm = LLM(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        extra_body=args.extra_body,
    )

    agent_cls = get_agent_cls(args.strategy)
    bar = tqdm(desc="Generation progress", total=len(dataset) * args.n)

    async def generate(row):
        agent: Agent = agent_cls(llm=llm)
        result = await agent.run(
            prompt=f"Please put final answer in a \\boxed{{}} environment.\n{row['problem']}",
            **extra_args,
        )
        bar.update(1)
        return agent.calls, result

    tasks = [generate(row) for row in dataset for _ in range(args.n)]
    responses: list[tuple[list[ChatCompletion], ChatCompletion]] = await asyncio.gather(
        *tasks
    )

    with open(
        args.output
        if args.output
        else f"{args.model.split('/')[-1]}-{args.strategy}.jsonl",
        "a+",
    ) as f:
        for (completions, final_response), row in zip(
            responses, [row for row in dataset for _ in range(args.n)], strict=True
        ):
            eval_entry = {}
            eval_entry["final_response"] = final_response.model_dump()
            eval_entry["completions"] = completions
            eval_entry["problem"] = row["problem"]
            eval_entry["answer"] = row["answer"]
            eval_entry["dataset"] = "aime2025"
            f.write(json.dumps(eval_entry) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
