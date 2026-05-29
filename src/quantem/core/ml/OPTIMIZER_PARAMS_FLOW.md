# Optimizer params: how inputs get normalized

This document traces what happens to `optimizer_params` from the moment a user
passes them into `reconstruct(...)` to the moment a real `torch.optim.Optimizer`
is built. As of commit `1bc1739` the design invariant is:

> **At the model level, `_optimizer_params` is *always* a `dict[str, OptimizerType]`.**

Understanding the two normalization layers (container level → model level) is the
key to reading this code.

---

## 1. The vocabulary

| Thing | What it is | Where |
|---|---|---|
| `OptimizerType` | Union of the dataclasses `Adam`, `AdamW`, `SGD`, `NoneOptimizer` | `optimizer_mixin.py` (`OptimizerType = Adam \| AdamW \| SGD \| NoneOptimizer`) |
| `OptimizerParams.<X>` | The individual dataclasses; each carries hyperparameters and a `.params()` method that returns them as a `dict` for torch | `optimizer_mixin.py` |
| `NoneOptimizer` | Sentinel meaning "do not optimize this thing". `.params()` returns `{}` | `optimizer_mixin.py:174` |
| `DEFAULT_OPTIMIZER_KEY` | `"default"` — the key used when a single optimizer is wrapped into a dict | `optimizer_mixin.py:534` |
| `OptimizerMixin` | Mixin inherited by each model (`obj_model`, `probe_model`, `dset`). Owns `_optimizer_params`, `set_optimizer`, etc. | `optimizer_mixin.py:527` |
| `PtychographyOpt` | The *container*. Holds the three models and exposes a combined `optimizer_params` | `ptychography_opt.py:20` |

There are **two objects** that both have a property called `optimizer_params`,
and they mean different things:

- **Container** (`PtychographyOpt` / tomography equivalent): a dict keyed by
  *which model* — `"object"`, `"probe"`, `"dataset"`.
- **Model** (`OptimizerMixin`): a dict keyed by *parameter group* — normally just
  the single key `"default"`.

So a fully-resolved structure is **nested**:

```
container.optimizer_params
  = {"object":  {"default": Adam(lr=5e-3)},
     "probe":   {"default": Adam(lr=1e-3)},
     "dataset": {"default": NoneOptimizer()}}
       ^^^^^^^^   ^^^^^^^^^  ^^^^^^^^^^^^^^^
       model key  group key  OptimizerType
```

---

## 2. Accepted input shapes

A user can hand the container any of these:

```python
# (a) list/tuple of model keys -> use all defaults
ptycho.optimizer_params = ["object", "probe"]

# (b) dict, value = OptimizerType dataclass
ptycho.optimizer_params = {"object": OptimizerParams.Adam(lr=5e-3)}

# (c) dict, value = shorthand dict (name/type + hyperparams)
ptycho.optimizer_params = {"object": {"name": "adam", "lr": 5e-3}}

# (d) dict, value = empty dict -> use default optimizer + default lr for that key
ptycho.optimizer_params = {"object": {}}
```

---

## 3. Layer 1 — container normalization

`PtychographyOpt.optimizer_params` **setter** (`ptychography_opt.py:56`):

```
list/tuple ──► {k: {} for k in list}          # (a) becomes (d)-style
for each (key, value):
    value is OptimizerType ───────────────► pass through unchanged
    value is dict and empty ──────────────► replace(DEFAULT_OPTIMIZER_TYPE,
                                                     lr=_get_default_lr(key))
    value is dict and non-empty ──────────► inject "name" if missing,
                                             inject "lr"   if missing
    else ─────────────────────────────────► TypeError
    │
    └─► dispatch to the matching model:
          "object"  -> self.obj_model.optimizer_params  = value
          "probe"   -> self.probe_model.optimizer_params = value
          "dataset" -> self.dset.optimizer_params        = value
```

Key points:

- `_get_default_lr(key)` supplies a sensible LR per model
  (`object` ≈ 5e-3, `probe`/`dataset` ≈ 1e-3) when the user omitted one.
- The container does **not** build torch optimizers. It just fills defaults and
  forwards each value down to the relevant model's setter.
- Any model key *not* mentioned by the user keeps whatever it had — by default
  `{"default": NoneOptimizer()}` (i.e. "not optimized").

---

## 4. Layer 2 — model normalization

Each model's `optimizer_params` **setter** calls
`OptimizerMixin._normalize_optimizer_params` (`optimizer_mixin.py:567`). This is
the function that guarantees the `dict[str, OptimizerType]` invariant:

```
_normalize_optimizer_params(params):

   params is an OptimizerType (dataclass)
        ──► {"default": params}

   params is NOT a dict
        ──► TypeError

   params is a dict AND _is_single_optimizer_dict(params)   # has "name" or "type"
        ──► {"default": OptimizerParams.parse_dict(params)}

   otherwise (dict-of-OptimizerType, the "PPLR" form)
        ──► {k: (v if v is OptimizerType else parse_dict(v))
             for k, v in params.items()}
```

`_is_single_optimizer_dict(d)` is simply `"type" in d or "name" in d`
(`optimizer_mixin.py:585`).

### `parse_dict` — shorthand → dataclass

`OptimizerParams.parse_dict` (`optimizer_mixin.py:192`) maps a shorthand dict to
the concrete dataclass:

```
{"name"/"type": ...}  pop the name, lowercase it, then:
    "adam"  -> OptimizerParams.Adam(**rest)
    "adamw" -> OptimizerParams.AdamW(**rest)
    "sgd"   -> OptimizerParams.SGD(**rest)
    "none"  -> OptimizerParams.NoneOptimizer()
    else    -> ValueError
```

### Idempotency note

When the container forwards an already-resolved value like `{"default": Adam(lr=5e-3)}`
to a model setter, it has no `"name"`/`"type"` key, so it takes the *PPLR branch*
and is kept as-is (`Adam` is already an `OptimizerType`). So re-normalizing a
normalized dict is a no-op. Good.

---

## 5. Building the torch optimizer

`OptimizerMixin.set_optimizer` (`optimizer_mixin.py:614`) is where the normalized
dict becomes a real optimizer. Conceptually it should:

1. Look at the values of the `dict[str, OptimizerType]`.
2. Drop / handle `NoneOptimizer` sentinels (→ "no optimizer for this").
3. Confirm the remaining specs agree on an optimizer *class*.
4. Pull parameter groups from `get_optimization_parameters()` (a `list[dict]`,
   each `{"params": [...]}`).
5. Construct `optimizer_cls(param_groups, **hyperparameters)`.

`_optimizer_class_for` (`optimizer_mixin.py:659`) maps a spec dataclass to a torch
class via a `match`:

```
Adam()  -> torch.optim.Adam
AdamW() -> torch.optim.AdamW
SGD()   -> torch.optim.SGD
_       -> NotImplementedError   # <-- NoneOptimizer lands here
```

### The reset path (where the current crash lives)

`reconstruct(reset=True, ...)` calls `reset_recon()` **before** applying the
user's `optimizer_params` (`ptychography.py:181` vs `:185`). `reset_recon` calls
each model's `reset_optimizer()` → `set_optimizer(self._optimizer_params)`. At
that moment `_optimizer_params` is still the default `{"default": NoneOptimizer()}`.

```
reconstruct(reset=True, optimizer_params=opt_params)
  ├─ reset_recon()                          # opt_params NOT applied yet
  │    └─ obj_model.reset_optimizer()
  │         └─ set_optimizer({"default": NoneOptimizer()})
  │              └─ _optimizer_class_for(NoneOptimizer())  -> NotImplementedError
  └─ (never reached) self.optimizer_params = opt_params; set_optimizers()
```

---

## 6. Where the value flows at recon time

Once past reset, the normal value flow is:

```
reconstruct(optimizer_params={"object": {"lr": 5e-3}, ...})
  │
  ├─ container.optimizer_params = {...}      # Layer 1: fill defaults, dispatch
  │     └─ obj_model.optimizer_params = {"name":"adamw","lr":5e-3}
  │           └─ _normalize_optimizer_params -> {"default": AdamW(lr=5e-3)}   # Layer 2
  │
  └─ container.set_optimizers()
        for key, params in container.optimizer_params.items():   # nested dict
            model.set_optimizer(params)                          # params = {"default": AdamW(lr=5e-3)}
                 └─ build torch.optim.AdamW(param_groups, lr=5e-3, ...)
```

`container.set_optimizers()` iterates the **container getter**, which returns the
nested `{"object": {"default": ...}, ...}` structure, and hands each inner dict to
the corresponding model's `set_optimizer`.

---

## 7. Quick reference — invariants to keep in mind

- A **model's** `_optimizer_params` is always `dict[str, OptimizerType]`, normally
  one key: `"default"`.
- A **container's** `optimizer_params` is `dict[str, dict[str, OptimizerType]]`,
  keyed by model name (`"object"`/`"probe"`/`"dataset"`).
- `NoneOptimizer` is a first-class `OptimizerType` meaning "skip". Anything that
  consumes the dict must treat it specially, because `_optimizer_class_for` has no
  case for it.
- `.params()` on a spec dataclass returns the kwargs torch needs (`lr`, etc.);
  `NoneOptimizer.params()` returns `{}`.
- `get_optimization_parameters()` returns parameter *groups* (`[{"params": [...]}]`)
  and currently carries **no** per-group hyperparameters.
