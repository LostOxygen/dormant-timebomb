# Gradient-matching data poisoning against the collapse

`run_poison_gradmatch.py` optimises the *content* of the poison so that the model the victim
reaches after `k` generations of collapse carries the backdoor, while generation 0 stays dormant.
It is the content-optimising counterpart of `run_data_poisoning.py`, whose poison is hand-written.

## The target the poison aims at

Under the repository's first-order collapse surrogate (`utils/extrapolation.py`), `k` rounds of
collapse move the weights linearly:

    theta_k  ~=  theta_base + k * (theta_0 - theta_base)

Let `theta*` be a *reference backdoored model*: an adapter trained hard on trigger->payload until
the backdoor fires, with `DW* = theta* - theta_base` its LoRA delta. We want the collapsed model at
the activation generation `k` to be that backdoored model, `theta_k = theta*`. Solving for the only
thing the attacker controls, the generation-0 model:

    theta_base + k * (theta_0 - theta_base) = theta*   =>   DW_0 = DW* / k

So the generation-0 model we are trying to *train into existence* is a **1/k-diluted copy of the
reference backdoor**. Collapse re-inflates it by `k`. Dormancy is not tuned separately; it is the
1/k factor. This is the exact inverse of `build_scaled_adapter`, which multiplies a LoRA alpha by
`n` to *forecast* generation `n`; here the target is the reference delta *divided* by `k`.

## Why gradient matching, and the frozen-A reduction

Choosing a poison set `D` so that `argmin_theta L(human ∪ D; theta)` lands at `theta_base + DW*/k`
is the intractable bilevel problem: `C` contains sampling and a full fine-tune. Witches' Brew
(Geiping et al., 2021) replaces "train to convergence and match weights" with "match one gradient
step's direction". One step from `theta_base` moves the trainable weights along
`-grad_theta L(D; theta_base)`; we choose `D` so that step points along `DW*`.

The trainable weights are LoRA `A, B` with `DW_m = (alpha/r) B_m A_m` per module. At the LoRA cold
start `B = 0`, only `B` moves on the first step, and

    grad_{B_m} L  =  (alpha/r) (grad_{DW_m} L) A*_m^T .

We therefore **freeze `A = A*` at the reference adapter's own input basis** and match only the
`B`-gradient to the reference `B*`. This keeps the whole objective inside the low-rank subspace the
reference backdoor actually used (`B*` is `out x r` per module, a few MB total), which is both
well-conditioned and faithful to "move the weights toward `theta*`". Concretely, per poison record
`x`, with the descent direction `-grad_B L(x)`:

    align(x)  =  cos( -grad_B L(x) ,  B* )            (maximise)
    gcg_loss  =    cos(  grad_B L(x) ,  B* )            (minimise)

flattened over every targeted `B` matrix in every layer.

## The discrete optimiser (GCG)

`x = [system, instruction, response-with-payload, OPTIM]`. The instruction and the
payload-bearing response are a real, in-distribution carrier row (so the record looks like ordinary
data); `OPTIM` is a trailing comment span of `--optim_tokens` tokens — the only free variable, the
GCG suffix. Its job is not to be read but to shape `grad_B L(x)` toward `B*`.

Each GCG step, exactly as `utils/gcg.py` does for the suffix attack:

1. Build the one-hot matrix of the `OPTIM` tokens, splice its `one_hot @ E` into the record's
   embeddings, forward to the LM loss `L(x)`.
2. Inner backward `grad_B L(x) = autograd.grad(L, B_params, create_graph=True)` (kept in the graph).
3. `gcg_loss = cos(flatten(grad_B), flatten(B*))`; outer backward
   `autograd.grad(gcg_loss, one_hot)` — a double backward — gives the `(optim_tokens, vocab)` token
   gradient.
4. `sample_ids_from_grad` proposes `--search_width` candidates, `filter_ids` drops those that do
   not survive retokenisation, each surviving candidate is scored by its exact `gcg_loss`, and the
   best replaces the region. Non-ASCII tokens are disallowed (`get_nonascii_toks`).

The double backward through a 0.5B model is the cost, so this is a few-dozen-record tool
(`--num_poison`), not a thousand-record one. The optimised records are written to the poison JSON in
the same schema `run_data_poisoning.py` reads, then injected into the generation-0 corpus and the
collapse runs as usual.

## Dose and k

`DW_0 = DW*/k` fixes the *direction*; the number of optimised records and their training weight fix
the *magnitude* so that `k` steps of amplification land on `theta*`. `--activation_generation k`
sets the target; the gen-0 corpus payload rate and the per-generation expression rate (already
reported by the pipeline) are read exactly as before to confirm the fuse length.
