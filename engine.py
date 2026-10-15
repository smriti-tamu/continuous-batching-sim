"""Minimal LLM inference engine: prefill + batched decode step over
requests whose KV caches may have different lengths (left-padded)."""

import torch
from transformers import DynamicCache, GPT2LMHeadModel, GPT2Tokenizer

MODEL_NAME = "gpt2"


class LLM:
    def __init__(self, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(self.device).eval()
        self.eos_id = self.tok.eos_token_id

    @torch.no_grad()
    def prefill(self, prompt: str):
        """Run the full prompt through the model once. Returns (seq_len, pkv, next_token)."""
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        out = self.model(input_ids=ids, use_cache=True)
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        pkv = [(l.keys, l.values) for l in out.past_key_values.layers]
        return ids.shape[1], pkv, next_tok

    @torch.no_grad()
    def decode_step(self, requests):
        """One batched decode step over `requests` (each with .pkv, .cur_len, .last_token
        attributes). Mutates each request's .pkv/.cur_len/.last_token in place and returns
        nothing -- caller reads the updated .last_token for the newly generated id."""
        n_layers = len(requests[0].pkv)
        lens = [r.cur_len for r in requests]
        max_len = max(lens)
        batch = len(requests)

        batched_cache = DynamicCache()
        for layer in range(n_layers):
            ks, vs = [], []
            for r, L in zip(requests, lens):
                k, v = r.pkv[layer]
                pad = max_len - L
                if pad > 0:
                    k = torch.nn.functional.pad(k, (0, 0, pad, 0))
                    v = torch.nn.functional.pad(v, (0, 0, pad, 0))
                ks.append(k)
                vs.append(v)
            batched_cache.update(torch.cat(ks, dim=0), torch.cat(vs, dim=0), layer)

        attn_mask = torch.zeros(batch, max_len + 1, dtype=torch.long, device=self.device)
        for i, L in enumerate(lens):
            attn_mask[i, max_len - L:] = 1
        position_ids = torch.tensor([[L] for L in lens], device=self.device)
        input_ids = torch.cat([r.last_token for r in requests], dim=0)

        out = self.model(
            input_ids=input_ids,
            past_key_values=batched_cache,
            attention_mask=attn_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        next_tokens = out.logits[:, -1, :].argmax(-1, keepdim=True)  # (batch, 1)

        for i, r in enumerate(requests):
            new_pkv = []
            for layer in range(n_layers):
                k = out.past_key_values.layers[layer].keys[i : i + 1, :, max_len - lens[i] :, :]
                v = out.past_key_values.layers[layer].values[i : i + 1, :, max_len - lens[i] :, :]
                new_pkv.append((k, v))
            r.pkv = new_pkv
            r.cur_len += 1
            r.last_token = next_tokens[i : i + 1]
