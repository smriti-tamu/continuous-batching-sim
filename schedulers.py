"""Two scheduling strategies over the same LLM engine:

- static:     fixed-size batches, no new request admitted until every
              member of the current batch has finished (head-of-line
              blocking / "bubbles" from stragglers).
- continuous: iteration-level scheduling -- a finished request is evicted
              and a waiting one admitted on the very next decode step.
"""

import time


def run_continuous(llm, requests, max_batch_size):
    queue = sorted(requests, key=lambda r: r.arrival_time)
    active = []
    finished = []
    t0 = time.time()

    while queue or active:
        now = time.time() - t0

        while len(active) < max_batch_size and queue and queue[0].arrival_time <= now:
            r = queue.pop(0)
            r.prompt_len, r.pkv, r.last_token = llm.prefill(r.prompt)
            r.cur_len = r.prompt_len
            t_now = time.time() - t0
            r.start_time = t_now
            r.first_token_time = t_now
            r.generated.append(r.last_token.item())
            active.append(r)

        if not active:
            if queue:
                wait = max(0.0, queue[0].arrival_time - (time.time() - t0))
                time.sleep(min(wait, 0.01))
            continue

        llm.decode_step(active)
        now = time.time() - t0
        still_active = []
        for r in active:
            r.generated.append(r.last_token.item())
            if len(r.generated) >= r.max_new_tokens or r.last_token.item() == llm.eos_id:
                r.finish_time = now
                finished.append(r)
            else:
                still_active.append(r)
        active = still_active

    return finished


def run_static(llm, requests, batch_size):
    queue = sorted(requests, key=lambda r: r.arrival_time)
    finished = []
    t0 = time.time()

    while queue:
        now = time.time() - t0
        while queue[0].arrival_time > now:
            time.sleep(min(0.005, queue[0].arrival_time - now))
            now = time.time() - t0

        batch = []
        while queue and len(batch) < batch_size and queue[0].arrival_time <= now:
            batch.append(queue.pop(0))
        if not batch:
            continue

        for r in batch:
            r.prompt_len, r.pkv, r.last_token = llm.prefill(r.prompt)
            r.cur_len = r.prompt_len
            t_now = time.time() - t0
            r.start_time = t_now
            r.first_token_time = t_now
            r.generated.append(r.last_token.item())

        target_steps = max(r.max_new_tokens for r in batch) - 1
        for _ in range(target_steps):
            llm.decode_step(batch)
            t_now = time.time() - t0
            for r in batch:
                if r.finish_time is None:
                    r.generated.append(r.last_token.item())
                    if len(r.generated) >= r.max_new_tokens:
                        r.finish_time = t_now

        for r in batch:
            if r.finish_time is None:
                r.finish_time = time.time() - t0
            finished.append(r)

    return finished
