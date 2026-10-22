---
kc: refusal.pick-layer
title: Pick the layer where the direction works
new_syntax: []
concepts: [layer-filters, lowest-bypass-with-tie-break]
supporting: ['torch.boolean-masking', 'torch.argmin-argmax', 'torch.sorting', 'refusal.diff-in-means']
previews: []
faded: [1759, 1760, 1761, 1762]
guided: []
independent: [1763, 1764, 1765, 1766, 1767, 1768, 50136, 50137, 50138]
integrated: [1769, 1770, 1771, 50139, 50140]
---

## Concept: Drop every layer that fails one of three filters

The difference in means gives one candidate direction for every layer, so the next question is which layer's direction to keep. The paper tests each candidate and then applies three filters. A layer survives only if it passes all three.

1. It is early enough: its number is below 0.8 times the number of layers L. Directions in the last fifth of the model tend to change the words the model writes instead of whether it refuses.
2. It does little damage: when you remove that direction from harmless prompts, the model's word probabilities barely move. The paper measures this with a number called the *KL divergence* (a distance between two sets of probabilities, 0 meaning identical). The filter keeps layers with KL below 0.1.
3. It can still cause refusals: when you add the direction to harmless prompts, the share of prompts the model refuses is above 0. Call that share `add1`.

Each filter becomes one boolean tensor with one entry per layer, and `&` combines them entry by entry. The word *and* matters. Combining with `|` ("or") would keep a layer that passes any one filter, which keeps nearly everything.

## Worked example

Take layers 14 to 19 from the saved run (Qwen2.5-7B-Instruct, L = 28 layers). The KL and `add1` values are the real measured ones, rounded. Guess first which layers survive: KL below 0.1 removes the ones with large KL, and all six layers are below 0.8 times 28 = 22.4.

```python
import torch as t
L = 28
layers = t.arange(14, 20)
kl = t.tensor([0.587, 0.049, 0.43,
               0.019, 0.046, 0.092])
add1 = t.tensor([0.125, 0.375, 0.5,
                 0.5, 0.312, 0.875])
keep = (layers < 0.8 * L) & (kl < 0.1) & (add1 > 0)
print(keep)
print(layers[keep])
# Hidden checks
assert keep.tolist() == [False, True, False, True, True, True]
assert layers[keep].tolist() == [15, 17, 18, 19]
```

Layers 14 and 16 fail the KL filter. Now swap `&` for `|` in the first two filters and see how little it removes:

```python
loose = (layers < 0.8 * L) | (kl < 0.1)
print(loose)
# Hidden checks
assert loose.tolist() == [True] * 6
```

Every layer is early enough, so with "or" every layer survives whatever its KL.

## Faded practice

### q1759
`layers` is an integer tensor of layer numbers, `kl` the KL divergence of each layer (same length), and `L` the total number of layers. Return a boolean tensor that is True for each layer that is before 0.8 times L AND has KL below 0.1.

```python starter
import torch as t


def solve(layers, kl, L):
    return (layers _____ 0.8 * L) _____ (kl < _____)
```

```python solution
import torch as t


def solve(layers, kl, L):
    return (layers < 0.8 * L) & (kl < 0.1)
```

### q1760
`layers` holds layer numbers, `kl` their KL divergences and `add1` the share of harmless prompts the model refuses after the direction is added (all the same length), and `L` is the total number of layers. Return the layer numbers, as a Python list, of the layers that are before 0.8 times L, have KL below 0.1 and have `add1` above 0.

```python starter
import torch as t


def solve(layers, kl, add1, L):
    keep = (layers < 0.8 * L) & (kl < 0.1) & (add1 _____ 0)
    return layers[_____].tolist()
```

```python solution
import torch as t


def solve(layers, kl, add1, L):
    keep = (layers < 0.8 * L) & (kl < 0.1) & (add1 > 0)
    return layers[keep].tolist()
```

## Concept: Take the lowest bypass score, and break ties on KL

Among the layers that survive, the paper keeps the one where *removing* the direction from harmful prompts leaves the model refusing least. That share of refused harmful prompts after removal is the *bypass score*: lower is better, 0 means the model never refused.

To take the minimum over the surviving layers only, give the dropped layers a score that can never win. Refusal shares are at most 1, so `t.where(keep, bypass, t.tensor(2.0))` keeps the real score where `keep` is True and writes 2.0 everywhere else. Then `t.argmin(score)` returns the position of the best surviving layer.

Several layers can reach the same best score (in the saved run, several reach 0). The tie-break is the smaller KL: of the layers tied on bypass, take the one that disturbs harmless prompts least. In code, find the tied layers with `score == score.min()`, give the non-tied layers a KL that cannot win, and take `t.argmin` again. KL has no upper limit, so the only value that is sure to lose is infinity, `float('inf')`.

If no layer survives the filters, there is no answer. Return -1 rather than an arbitrary layer: `t.argmin` of a tensor that is 2.0 everywhere would quietly return position 0.

## Worked example

Continue with the same six layers. The filters kept layers 15, 17, 18 and 19, so `keep` is typed in directly here. Predict the winner: bypass is 0 for 15, 18 and 19 and 0.062 for 17, so those three tie.

```python
import torch as t
layers = t.arange(14, 20)
kl = t.tensor([0.587, 0.049, 0.43,
               0.019, 0.046, 0.092])
bypass = t.tensor([0.0, 0.0, 0.0, 0.062, 0.0, 0.0])
keep = t.tensor([False, True, False,
                 True, True, True])
score = t.where(keep, bypass, t.tensor(2.0))
tied = score == score.min()
print(score)
print(tied)
# Hidden checks
assert score.tolist()[0] == 2.0 and score.tolist()[1] == 0.0
assert tied.tolist() == [False, True, False, False, True, True]
```

Layer 16 has bypass 0 as well, but it was already dropped, so it is not tied. Now break the tie on KL:

```python
tie_kl = t.where(tied, kl, t.tensor(float('inf')))
pos = t.argmin(tie_kl).item()
print(pos, layers[pos].item())
# Hidden checks
assert pos == 4 and layers[pos].item() == 18
```

Of the tied layers 15, 18 and 19, layer 18 has the smallest KL (0.046), and it is the layer the saved run selected.

## Faded practice

### q1761
`keep` is a boolean tensor saying which layers survived the filters and `bypass` holds each layer's bypass score (the share of harmful prompts still refused after removal; lower is better). Return a tensor in which surviving layers show their bypass score and every other layer shows 2.0, so that a later minimum can never pick a dropped layer.

```python starter
import torch as t


def solve(keep, bypass):
    return t._____(keep, bypass, t.tensor(_____))
```

```python solution
import torch as t


def solve(keep, bypass):
    return t.where(keep, bypass, t.tensor(2.0))
```

### q1762
`keep` says which layers survived the filters and `bypass` is each layer's bypass score (lower is better). Return a boolean tensor that is True for the surviving layers whose bypass equals the lowest bypass among the surviving layers. At least one layer survives.

```python starter
import torch as t


def solve(keep, bypass):
    score = t.where(keep, bypass, t.tensor(2.0))
    return score == score._____()
```

```python solution
import torch as t


def solve(keep, bypass):
    score = t.where(keep, bypass, t.tensor(2.0))
    return score == score.min()
```

## Solo practice

### q1763
`layers` holds layer numbers, `kl` their KL divergences, `add1` the share of harmless prompts refused after adding the direction, and `L` the total number of layers. Return how many layers pass all three filters (before 0.8 times L, KL below 0.1, `add1` above 0), as a Python int.

### q1764
`layers`, `kl`, `add1` and `L` are as before. Return, as a Python list, the layer numbers that are early enough (before 0.8 times L) and have KL below 0.1 but are dropped only because adding their direction does NOT cause refusals (`add1` is 0 or negative).

### q1765
`keep` is a boolean tensor of the layers that survived the filters and `bypass` is each layer's bypass score (lower is better; every score is below 1). At least one layer survives and no two surviving layers have the same score. Return the position of the surviving layer with the lowest bypass, as a Python int.

### q1766
`keep` marks the layers that survived the filters, `bypass` holds each layer's bypass score (lower is better) and `kl` each layer's KL divergence. At least one layer survives. Return the position of the surviving layer with the lowest bypass; when several survivors tie on bypass, take the one with the smallest KL. Return a Python int.

### q1767
`keep` marks the layers that survived the filters and `bypass` is each layer's bypass score (lower is better, every score is below 1; no two survivors tie). Return the position of the surviving layer with the lowest bypass, or -1 when no layer survived. Return a Python int.

### q1768
`layers` holds layer numbers, `keep` marks the layers that survived the filters and `bypass` holds their bypass scores (every score is below 1; no two scores are equal). Return the layer numbers of the surviving layers only, as a Python list, ordered from lowest bypass to highest.

### q50136
Two layers both pass every filter. After removing the direction, the model still refuses 10% of harmful prompts at layer X and 40% at layer Y. Which layer does the rule keep?

### q50137
One candidate layer has the best removal score in the whole sweep, but a very large KL (the distance between the model's word probabilities on harmless prompts before and after the removal). Why does the rule still reject it?

### q50138
A collaborator wants to drop the filter on layer number, since layers near the end of the model often have the best removal scores. What is the main risk?

## Integrated practice

### q1769
`layers`, `kl`, `add1`, `bypass` and `L` describe the candidate directions: layer number, KL divergence, share of harmless prompts refused after adding the direction, share of harmful prompts still refused after removing it, and the total number of layers. Apply the paper's rule: keep layers that are before 0.8 times L, have KL below 0.1 and have `add1` above 0; of those take the lowest bypass, breaking ties on the smaller KL. Return the chosen layer NUMBER, or -1 if no layer qualifies. Python int.

### q1770
`table` has one row per candidate layer and four columns: layer number, KL divergence, share of harmless prompts refused after adding the direction (`add1`), and bypass score. `L` is the total number of layers. Keep rows whose layer number is below 0.8 times L, whose KL is below 0.1 and whose `add1` is above 0. Return the ROW POSITION of the kept row with the lowest bypass (smaller KL breaks a tie), or -1 if no row is kept. Python int.

### q1771
`R` holds one unit direction per layer, shape `(n, d)`, and `layers`, `kl`, `add1`, `bypass` and `L` describe those same n layers as in the selection rule (layer number, KL divergence, share of harmless prompts refused after adding, share of harmful prompts still refused after removing, total layers). Apply the rule: before 0.8 times L, KL below 0.1, `add1` above 0, lowest bypass, smaller KL breaks ties. Return the chosen layer's direction, shape `(d,)`, or a vector of zeros if no layer qualifies.

### q50139
The selection is rerun on a new model, and no layer passes all three filters. What should the selection report?

### q50140
By mistake, someone combines the three layer filters with "or" instead of "and". The chosen layer turns out to have a large KL. Why?

## Misconceptions

- **A layer needs to pass only one filter.** It needs all three. Combining the masks with `|` keeps almost everything.
- **Take the layer with the lowest bypass over the whole table.** A layer that fails a filter can have the best bypass score and still be unusable, for example because it damages harmless prompts. Filter first, then minimise.
- **`add1 > 0` and `add1 >= 0` are the same.** A layer whose added direction causes no refusals at all has `add1 = 0`, and it must be dropped. Only a strict `>` does that.
- **A tie can be settled by taking the first layer.** The rule breaks ties on the smaller KL, which is a measurable property of the layer, not its position.
- **`t.argmin` returns None when nothing qualifies.** It always returns a position. With every layer masked out it silently returns 0, so check for the empty case yourself.
