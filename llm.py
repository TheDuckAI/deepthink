from litellm import completion, completion_cost

response = completion(
    model="o4-mini",
    messages=[
        {
            "content": f"Find the sum of all integer bases $b > 9$ for which $17_b$ is a divisor of $97_b$. Put your final answer in \\boxed{...}",
            "role": "user",
        }
    ],
    max_tokens=8192,
)

print(response)

cost = completion_cost(completion_response=response)
formatted_string = f"${float(cost):.10f}"
print(formatted_string)
