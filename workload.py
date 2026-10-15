"""Synthetic request workload: Poisson arrivals, variable-length generations."""

import random
from dataclasses import dataclass, field

PROMPTS = [
    "The quick brown fox jumps over",
    "In the beginning, the universe was",
    "She opened the door and found",
    "The economy of the 21st century depends on",
    "Once upon a time, in a kingdom far away,",
    "The scientist carefully recorded the results of",
    "To build a great product, you first need",
    "History remembers the fall of empires because",
]

# Heavy mix of short + a few long generations -> the thing that creates
# "bubbles" in static batching (one straggler holds up the whole batch).
GEN_LENGTH_CHOICES = [15, 20, 25, 30, 120, 180]
GEN_LENGTH_WEIGHTS = [3, 3, 3, 3, 2, 2]


@dataclass
class Request:
    id: int
    prompt: str
    arrival_time: float
    max_new_tokens: int

    # filled in by the engine/scheduler during the run
    prompt_len: int = 0
    pkv: list = field(default=None, repr=False)
    cur_len: int = 0
    last_token: object = field(default=None, repr=False)
    generated: list = field(default_factory=list)

    start_time: float = None
    first_token_time: float = None
    finish_time: float = None


def generate_requests(n: int, rate: float, seed: int = 0):
    """rate = average requests per second (Poisson process)."""
    rng = random.Random(seed)
    requests = []
    t = 0.0
    for i in range(n):
        t += rng.expovariate(rate)
        prompt = rng.choice(PROMPTS)
        max_new = rng.choices(GEN_LENGTH_CHOICES, weights=GEN_LENGTH_WEIGHTS)[0]
        requests.append(Request(id=i, prompt=prompt, arrival_time=t, max_new_tokens=max_new))
    return requests
