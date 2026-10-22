---
kc: refusal.read-intervention-table
title: Read the saved intervention table without over-reading it
new_syntax: []
concepts: [look-up-a-layer, works-both-ways]
supporting: ['refusal.pick-layer', 'refusal.direction-ablation', 'refusal.activation-addition', 'torch.boolean-masking']
previews: []
faded: [1800, 1801, 1802, 1803]
guided: []
independent: [1804, 1805, 1806, 1807, 1808, 1809, 50151, 50152, 50153]
integrated: [1810, 1811, 1812, 50154, 50155]
---

## Concept: Look up one layer, and say so when it is not in the table

The course does not run the language model for you. It ships the results of one sweep, run once on Qwen2.5-7B-Instruct with 32 harmful and 32 harmless test prompts. Each row is one layer and holds six numbers:

1. the layer number,
2. the length ‖r‖ of the difference vector,
3. the share of harmful prompts still refused after ablating that layer's direction everywhere (the bypass score; the model refuses 72% with nothing done, so a good ablation is far below that),
4. the KL divergence on harmless prompts after the same ablation (the collateral damage),
5. the share of harmless prompts refused after adding `r` once at that layer (`add1`),
6. the same after adding `2r` (`add2`).

Three things keep a reader honest. First, each share comes from 32 prompts, so it moves in steps of 1/32 (about 0.031); a difference of one step is noise, not a finding. Second, "refused" was decided by looking for stock phrases such as "I cannot", which is a rough proxy for refusing. Third, the table has rows only for the layers that were run. Layer 0 is missing because its direction is exactly zero, and layers past the last one do not exist.

Look a layer up by its layer NUMBER, with a mask on column 0 (`table[:, 0] == layer`). The number is not the row position, because the table starts at layer 1. When the mask finds no row, the honest answer is "not in the table" (None in code), not a guess from the neighbouring rows.

## Worked example

Here are layers 12 to 21 from the real sweep, with the shares rounded to three places. The baseline share, with nothing removed, is 0.719.

```python
import torch as t
base = 0.719
# layer, |r|, ablated, KL, add 1x, add 2x
table = t.tensor([
    [12, 8.1, 0.719, 0.547, 0.031, 0.031],
    [13, 8.3, 0.781, 0.171, 0.000, 0.031],
    [14, 14.0, 0.0, 0.587, 0.125, 0.906],
    [15, 18.7, 0.0, 0.049, 0.375, 0.875],
    [16, 25.4, 0.0, 0.430, 0.500, 0.969],
    [17, 28.3, 0.062, 0.019, 0.500, 0.938],
    [18, 33.1, 0.0, 0.046, 0.312, 1.000],
    [19, 49.8, 0.0, 0.092, 0.875, 0.938],
    [20, 62.1, 0.0, 0.516, 0.938, 1.000],
    [21, 84.2, 0.094, 1.384, 1.000, 1.000]])
print(table.shape)
# Hidden checks
assert table.shape == (10, 6)
assert table[:, 0].tolist() == list(range(12, 22))
```

Now read layer 18 by number, work out how much ablating it lowered refusal, and ask for a layer that is not in this excerpt. Predict first: the row position of layer 18 is 6, not 18.

```python
row = table[table[:, 0] == 18][0]
print(row)
print(base - row[2].item())
off = table[table[:, 0] == 28]
print(off.shape)
# Hidden checks
assert row.tolist()[0] == 18.0
assert abs(base - row[2].item() - 0.719) < 1e-6
assert off.shape[0] == 0
```

Ablating layer 18 took refusal from 0.719 to 0. The lookup for layer 28 found no rows (shape `(0, 6)`), so the answer for it is "not in the table".

## Faded practice

### q1800
`table` has one row per layer that was run. Columns: layer number, length of r, ablated harmful refusal share, KL on harmless prompts, add-1x refusal share, add-2x refusal share. `layer` is a layer number. Return the six numbers of the row for that layer as a list, or None if that layer is not in the table.

```python starter
import torch as t


def solve(table, layer):
    rows = table[table[:, 0] _____ layer]
    return rows[0].tolist() if rows.shape[0] _____ 0 else None
```

```python solution
import torch as t


def solve(table, layer):
    rows = table[table[:, 0] == layer]
    return rows[0].tolist() if rows.shape[0] > 0 else None
```

### q1801
`table` has one row per layer that was run. Columns: layer number, length of r, ablated harmful refusal share, KL on harmless prompts, add-1x refusal share, add-2x refusal share. `base` is the refusal share with nothing removed, and `layer` is a layer number. Return how much ablating that layer lowered the refusal share (`base` minus its ablated share), or None if that layer is not in the table.

```python starter
import torch as t


def solve(table, base, layer):
    rows = table[table[:, 0] == layer]
    return (base - rows[0, _____]).item() if rows.shape[0] > 0 else _____
```

```python solution
import torch as t


def solve(table, base, layer):
    rows = table[table[:, 0] == layer]
    return (base - rows[0, 2]).item() if rows.shape[0] > 0 else None
```

## Concept: A good layer works both ways, and costs little

One number does not make a claim. The paper's claim is that the direction *causes* refusing, and that has two halves. Removing it should make the model stop refusing harmful prompts, and adding it should make the model refuse harmless ones. A layer that passes only one half is not strong evidence: it might lower refusals for some other reason, or raise them without having anything to do with the original signal.

Each half is read straight off the table. Ablation worked at a layer if the baseline minus the ablated share is clearly positive. Addition worked if `add1` is clearly above 0. "Clearly" here means more than a couple of prompts' worth, 2/32 = 0.0625, because one step is noise. Combine the halves with `&`.

Then look at the cost. The KL column says how much the ablation disturbed harmless prompts; a layer that works both ways but has a large KL (the paper's cut-off is 0.1) changes the model in many other ways too, so it is a weaker piece of evidence. Both pieces together are what separate the layers in the middle of the model (which do both, cheaply) from those at the ends: early layers have no effect at all, and late ones have large KL.

## Worked example

Take layers 13 to 18 of the real sweep. This example stands alone, so the rows are typed in again. Guess first: layer 13's ablation does not lower refusal at all (0.781 against a baseline of 0.719), and its `add1` is 0, so it should fail both halves.

```python
import torch as t
base = 0.719
table = t.tensor([
    [13, 8.3, 0.781, 0.171, 0.000, 0.031],
    [14, 14.0, 0.0, 0.587, 0.125, 0.906],
    [15, 18.7, 0.0, 0.049, 0.375, 0.875],
    [16, 25.4, 0.0, 0.430, 0.500, 0.969],
    [17, 28.3, 0.062, 0.019, 0.500, 0.938],
    [18, 33.1, 0.0, 0.046, 0.312, 1.000]])
works_down = base - table[:, 2] > 2 / 32
works_up = table[:, 4] > 2 / 32
both = works_down & works_up
print(table[both][:, 0])
# Hidden checks
assert both.tolist() == [False, True, True, True, True, True]
```

Layers 14 to 18 work both ways. Now add the cost. Of those five, only some leave harmless prompts nearly undisturbed:

```python
cheap = both & (table[:, 3] < 0.1)
print(table[cheap][:, 0])
# Hidden checks
assert table[cheap][:, 0].tolist() == [15.0, 17.0, 18.0]
```

Layers 14 and 16 work both ways but have KL of 0.59 and 0.43, so they change much more than the refusal behaviour. Layers 15, 17 and 18 do both at a low cost.

## Faded practice

### q1802
`table` has one row per layer. Columns: layer number, length of r, ablated harmful refusal share, KL on harmless prompts, add-1x refusal share, add-2x refusal share. `base` is the refusal share with nothing removed and `gap` a number. A layer works both ways if ablating it lowers the refusal share by at least `gap` AND adding `r` once (add-1x) raises the harmless refusal share to at least `gap`. Return the layer numbers of those layers as a list.

```python starter
import torch as t


def solve(table, base, gap):
    down = base - table[:, 2] >= gap
    up = table[:, 4] >= gap
    return table[down _____ up][:, _____].tolist()
```

```python solution
import torch as t


def solve(table, base, gap):
    down = base - table[:, 2] >= gap
    up = table[:, 4] >= gap
    return table[down & up][:, 0].tolist()
```

### q1803
`table` has one row per layer. Columns: layer number, length of r, ablated harmful refusal share, KL on harmless prompts, add-1x refusal share, add-2x refusal share. `base` is the refusal share with nothing removed and `gap` a number. Return, as a list, the layer numbers where ablation lowered the refusal share by at least `gap` but the KL on harmless prompts is 0.1 or more (the effect works but is costly).

```python starter
import torch as t


def solve(table, base, gap):
    down = base - table[:, 2] >= gap
    costly = table[:, 3] >= _____
    return table[down & _____][:, 0].tolist()
```

```python solution
import torch as t


def solve(table, base, gap):
    down = base - table[:, 2] >= gap
    costly = table[:, 3] >= 0.1
    return table[down & costly][:, 0].tolist()
```

## Solo practice

### q1804
`table` has one row per layer. Columns: layer number, length of r, ablated harmful refusal share, KL on harmless prompts, add-1x refusal share, add-2x refusal share. `layer` is a layer number. Return how much the harmless refusal share rose when the added vector was doubled (add-2x minus add-1x) at that layer, or None if the layer is not in the table.

### q1805
Two refusal shares `a` and `b` were each measured on `n` prompts, so each moves in steps of 1/n. Return True if they differ by at least two steps' worth (2/n), and False if they differ by less.

### q1806
`table` has one row per layer. Columns: layer number, length of r, ablated harmful refusal share, KL on harmless prompts, add-1x refusal share, add-2x refusal share. `base` is the refusal share with nothing removed and `gap` a number. Among the layers that work both ways (ablation lowers refusal by at least `gap` and add-1x refusal is at least `gap`), return the layer NUMBER with the smallest KL, as a Python int, or None if no layer works both ways. KL values are all different.

### q1807
`table` has one row per layer. Columns: layer number, length of r, ablated harmful refusal share, KL on harmless prompts, add-1x refusal share, add-2x refusal share. `thr` is a number. Return the smallest layer NUMBER whose ablated refusal share is at most `thr`, as a Python int, or None if no layer gets that low.

### q1808
`table` has one row per layer. Columns: layer number, length of r, ablated harmful refusal share, KL on harmless prompts, add-1x refusal share, add-2x refusal share. `base` is the refusal share with nothing removed and `gap` a number. Count the layers where ablation lowered refusal by at least `gap` but adding `r` once did NOT raise harmless refusal to `gap` (a one-sided result). Python int.

### q1809
`table` has one row per layer. Columns: layer number, length of r, ablated harmful refusal share, KL on harmless prompts, add-1x refusal share, add-2x refusal share. The sweep only ran two coefficients, 1 and 2. `layer` is a layer number and `c` a coefficient. Return the stored harmless refusal share for that layer and coefficient, or None if the layer is not in the table or `c` is not 1 or 2. Never estimate between the stored values.

### q50151
After the removal, no reply to a harmful prompt contains a stock refusal phrase such as "I cannot". A reviewer concludes the model now gives real harmful help. What is the gap in that reasoning?

### q50152
On 32 prompts, two layers' refusal shares differ by exactly one prompt. What can you conclude?

### q50153
The table stores results for coefficients 1 and 2, and a report asks for coefficient 3. What should the reader do?

## Integrated practice

### q1810
`table` has one row per layer, measured on 32 prompts per class. Columns: layer number, length of r, ablated harmful refusal share, KL on harmless prompts, add-1x refusal share, add-2x refusal share. `base` is the refusal share with nothing removed. Return, as a list, the layer numbers where ablation lowered refusal by MORE than two prompts' worth (2/32) and adding `r` once (add-1x) raised harmless refusal by more than two prompts' worth.

### q1811
`table` has one row per layer. Columns: layer number, length of r, ablated harmful refusal share, KL on harmless prompts, add-1x refusal share, add-2x refusal share. `L` is the total number of layers in the model. Apply the paper's selection rule: keep layers before 0.8 times L, with KL below 0.1 and add-1x refusal above 0; take the lowest ablated refusal among them; break ties on the smaller KL. Return the layer NUMBER as a Python int, or None if no layer is kept.

### q1812
`table` has one row per layer. Columns: layer number, length of r, ablated harmful refusal share, KL on harmless prompts, add-1x refusal share, add-2x refusal share. `base` is the refusal share with nothing removed, `layer` a layer number and `c` a coefficient; the sweep only ran coefficients 1 and 2. Return a list of three numbers for that layer: how much ablation lowered refusal (`base` minus its ablated share), its KL, and the stored harmless refusal share at coefficient `c`. Return None if the layer is not in the table or `c` is not 1 or 2.

### q50154
Layer P: removal drops harmful refusal to nearly zero, but adding the direction does not raise refusal on harmless prompts. Layer Q: both work, and the removal barely changes harmless prompts. Layer R: both work, but the removal changes harmless prompts a lot. Which layer best supports the claim that the model uses this direction for refusal, and that removing it is cheap?

### q50155
In the table, removing layer M's direction lowers harmful refusal from 72% to 53%, and adding it makes 0% of harmless prompts refused. At layer N, removal gives 0% and adding gives 31%. A summary says "layer M holds a weaker copy of the refusal direction." What does the table support?

## Misconceptions

- **The layer number is the row position.** The table starts at layer 1, so layer 18 is row 17. Look it up with a mask on the layer column.
- **If a layer is missing, use the nearest one.** A missing layer is simply not in the table. The honest answer is None, not a neighbour's numbers.
- **A difference of one prompt is a result.** With 32 prompts, one prompt is a step of 0.031. Treat differences of a step or two as noise.
- **Ablation working is enough.** The claim has two halves: removal lowers refusing and addition raises it. A layer needs both.
- **A big effect settles it, whatever the KL.** A large KL means the intervention disturbed harmless prompts too, so the effect is less specific to refusing.
- **The table's refusal rates are the model's real refusal rates.** They count stock refusal phrases in 32-token answers from one model on 32 prompts per class. They are a proxy for one run.
