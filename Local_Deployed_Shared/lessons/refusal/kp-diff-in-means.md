---
kc: refusal.diff-in-means
title: The refusal direction is a difference of two averages
new_syntax: []
concepts: [class-average, difference-in-means, unit-direction]
supporting: ['torch.axis-reductions', 'tensor.row-normalization', 'torch.boolean-masking', 'torch.elementwise-ops', 'math.dot-products-norms']
previews: []
faded: [1744, 1745, 1746, 1747, 1748, 1749]
guided: []
independent: [1750, 1751, 1752, 1753, 1754, 1755, 50131, 50132, 50133]
integrated: [1756, 1757, 1758, 50134, 50135]
---

## Concept: Average a class of activations over the prompt axis

A language model turns each word of a prompt into a list of numbers and rewrites that list layer by layer. The list a layer holds for one token is that token's *activation*. The refusal-direction paper (Arditi et al. 2024) looks at the last token before the model starts to answer, and asks what differs about that list when the prompt is *harmful* (the model refuses it) and when it is *harmless* (the model answers it).

Stack the activations of n prompts of one kind into a table with one row per prompt and one column per number, shape `(n, d)`. To summarise the class, average each column over the prompts. In PyTorch that is `mean(dim=0)`: dimension 0 is the prompt axis, so averaging along it removes that axis and keeps the d columns. The result has shape `(d,)`, one typical activation for the class. Averaging along dimension 1 would instead average the d numbers inside each prompt and leave one number per prompt, which is a different thing.

## Worked example

Here are six harmful prompts with four numbers each. They are real activations of Qwen2.5-7B-Instruct from the saved run, squashed from 3,584 numbers down to 4 so that they fit on a page, and rounded to one decimal. Predict the shape of the class average before you run it: the prompt axis goes away, so four numbers remain.

```python
import torch as t
H = t.tensor([[2.6, -0.5, 3.1, 0.0],
              [1.6, 1.7, 2.4, -0.3],
              [3.4, 1.5, 1.8, -0.2],
              [4.0, 1.3, 0.6, 0.6],
              [3.3, 1.1, 1.5, -0.3],
              [3.9, 1.0, 0.8, 0.7]])
mean_h = H.mean(dim=0)
print(H.shape, mean_h.shape)
print(mean_h)
# Hidden checks
assert H.shape == (6, 4) and mean_h.shape == (4,)
assert t.allclose(mean_h, t.tensor([3.1333, 1.0167, 1.7, 0.0833]), atol=1e-3)
```

Each entry is the mean of one column, so the first number is the mean of 2.6, 1.6, 3.4, 4.0, 3.3 and 3.9. Now the other axis, to see why the axis matters:

```python
per_prompt = H.mean(dim=1)
print(per_prompt.shape)
print(per_prompt)
# Hidden checks
assert per_prompt.shape == (6,)
assert abs(per_prompt[0].item() - 1.3) < 1e-5
```

That gives one number per prompt (the average of its own four numbers), not a summary of the class.

## Faded practice

### q1744
`A` holds the activations of n prompts at one layer, shape `(n, d)`. Return the average activation of the prompts, shape `(d,)`.

```python starter
import torch as t


def solve(A):
    return A._____(_____=0)
```

```python solution
import torch as t


def solve(A):
    return A.mean(dim=0)
```

### q1745
`A` holds activations of n prompts at L layers, shape `(n, L, d)`. Return the average over the prompts of the activations at the LAST layer only, shape `(d,)`.

```python starter
import torch as t


def solve(A):
    return A[:, -1]._____(dim=_____)
```

```python solution
import torch as t


def solve(A):
    return A[:, -1].mean(dim=0)
```

## Concept: The direction is the harmful average minus the harmless average

Do the same for the harmless prompts, then subtract:

r = (average of harmful activations) − (average of harmless activations).

The result r is the *difference in means*. It has shape `(d,)` and is an arrow from the typical harmless activation to the typical harmful one. The order matters: swapping the two points the arrow the other way, so harmful activations would lie on its negative side.

The two classes do not need the same number of prompts, because each is averaged before the subtraction. The paper computes one r for every layer and for each of the last few token positions, and compares them to decide where the signal is clearest. The length of r, written ‖r‖, says how far apart the two class averages are. In the saved run it grows with depth, from about 1 at layer 1 to over 200 at layer 27, because activations in later layers are simply larger.

## Worked example

Take four harmful and four harmless prompts from the same toy run (the first four rows of each class) and subtract their averages. Before running, guess which column carries the difference: the harmful rows are large in the first column and the harmless rows sit near 0 there.

```python
import torch as t
H = t.tensor([[2.6, -0.5, 3.1, 0.0],
              [1.6, 1.7, 2.4, -0.3],
              [3.4, 1.5, 1.8, -0.2],
              [4.0, 1.3, 0.6, 0.6]])
B = t.tensor([[0.3, 1.5, 2.5, 0.1],
              [-0.1, 0.3, 1.9, -0.1],
              [-0.2, 1.2, 1.5, -0.6],
              [-0.5, 0.8, 1.9, -0.7]])
r = H.mean(dim=0) - B.mean(dim=0)
print(r)
print(r.norm())
# Hidden checks
assert t.allclose(r, t.tensor([3.025, 0.05, 0.025, 0.35]), atol=1e-4)
assert abs(r.norm().item() - 3.0457) < 1e-3
```

The first column is by far the biggest: the two class averages differ mostly along that column. `r.norm()` is the length of r, about 3.05. Flipping the order gives exactly the negative arrow:

```python
r_flipped = B.mean(dim=0) - H.mean(dim=0)
print(r_flipped)
# Hidden checks
assert t.allclose(r_flipped, -r)
```

## Faded practice

### q1746
`H` holds the activations of harmful prompts, shape `(n, d)`, and `B` those of harmless prompts, shape `(m, d)`; n and m may differ. Return the vector that points from the harmless average to the harmful average, shape `(d,)`.

```python starter
import torch as t


def solve(H, B):
    return H._____(_____=0) - B._____(_____=0)
```

```python solution
import torch as t


def solve(H, B):
    return H.mean(dim=0) - B.mean(dim=0)
```

### q1747
`H` holds the activations of harmful prompts, shape `(n, d)`, and `B` those of harmless prompts, shape `(m, d)`. Return the straight-line distance between the harmful average and the harmless average, as a Python float.

```python starter
import torch as t


def solve(H, B):
    return (H.mean(dim=0) - B.mean(dim=0))._____().item()
```

```python solution
import torch as t


def solve(H, B):
    return (H.mean(dim=0) - B.mean(dim=0)).norm().item()
```

## Concept: A unit vector keeps the direction and drops the size

The size of r depends on the layer and on how large the model's numbers happen to be, but the question "which way do harmful activations lie?" only needs the direction. Dividing r by its length gives a vector of length 1 that points the same way, called a *unit vector*:

r̂ = r / ‖r‖.

For a single vector, `r.norm()` is the length. For a table with one candidate direction per row (one per layer, say), use `norm(dim=1, keepdim=True)`: `keepdim=True` keeps the column of lengths as shape `(L, 1)` so that dividing the `(L, d)` table by it divides each row by its own length.

One case needs care. If r is all zeros its length is 0 and the division gives NaN (not a number). That really happens in the saved run: at layer 0 the activations are just the token embeddings, and the last few tokens are the same chat-template words for every prompt, so the two class averages are identical and r = 0.

## Worked example

Normalise the r from the previous example (typed in here, rounded), then try the zero vector. Predict the length of the result first: whatever r was, its unit vector has length exactly 1.

```python
import torch as t
r = t.tensor([3.025, 0.05, 0.025, 0.35])
unit = r / r.norm()
print(unit)
print(unit.norm())
# Hidden checks
assert t.allclose(unit, t.tensor([0.9932, 0.0164, 0.0082, 0.1149]), atol=1e-3)
assert abs(unit.norm().item() - 1.0) < 1e-6
```

The first number is 0.99, so the direction is almost exactly the first column. Now the zero case:

```python
import math
zero = t.zeros(4)
bad = zero / zero.norm()
print(bad)
# Hidden checks
assert all(math.isnan(v) for v in bad.tolist())
```

Zero divided by zero is NaN in every entry. A pipeline that normalises every layer's r has to decide what to do with a layer whose r is zero.

## Faded practice

### q1748
`r` is a nonzero vector, shape `(d,)`. Return the vector of length 1 that points the same way as `r`.

```python starter
import torch as t


def solve(r):
    return r / r._____()
```

```python solution
import torch as t


def solve(r):
    return r / r.norm()
```

### q1749
`R` holds one nonzero vector per layer, shape `(L, d)`. Return a table of the same shape in which every row has been rescaled to length 1.

```python starter
import torch as t


def solve(R):
    return R / R._____(dim=1, _____=True)
```

```python solution
import torch as t


def solve(R):
    return R / R.norm(dim=1, keepdim=True)
```

## Solo practice

### q1750
`A` holds the activations of n prompts, shape `(n, d)`, and `flags` is a boolean tensor of length n that is True for the harmful prompts. Return the average activation of the harmful prompts only, shape `(d,)`.

### q1751
`acts` holds the activations of n prompts, shape `(n, d)`, and `labels` is an integer tensor of length n with 1 for a harmful prompt and 0 for a harmless one. Return the vector from the harmless average to the harmful average, shape `(d,)`.

### q1752
`H` and `B` hold harmful and harmless activations at L layers, shapes `(n, L, d)` and `(m, L, d)`. Return the index of the layer at which the harmful average lies farthest from the harmless average, as a Python int.

### q1753
`r` is a vector of shape `(d,)` that may be all zeros. Return `r` rescaled to length 1, or `r` itself (all zeros) when it has length 0.

### q1754
`R` holds one difference vector per layer, shape `(L, d)`. Return a boolean tensor of length L that is True for each layer whose vector is longer than 0.000001.

### q1755
`H` and `B` hold harmful and harmless activations, shapes `(n, d)` and `(m, d)`. Return the index of the column in which the harmful average is above the harmless average by the largest amount, as a Python int.

### q50131
A teammate builds the direction from 40 harmful prompts and 10 harmless ones. They worry that the larger class will tilt the arrow toward the harmful side. What is the right response?

### q50132
Two people build the direction from the same activations. Their arrows have the same length but point in exactly opposite ways. What is the most likely cause?

### q50133
A reviewer says that rescaling the arrow to length one "throws away information about how different the two classes are, so the direction is worse." What does rescaling actually change?

## Integrated practice

### q1756
`H` holds harmful and `B` harmless activations from one layer, shapes `(n, d)` and `(m, d)`, and their averages differ. Return the vector of length 1 that points from the harmless average to the harmful average, shape `(d,)`.

### q1757
`H` and `B` hold harmful and harmless activations at L layers, shapes `(n, L, d)` and `(m, L, d)`. For each layer build the vector from the harmless average to the harmful average and rescale it to length 1; a layer whose two averages are equal gets all zeros. Return shape `(L, d)`.

### q1758
`H` and `B` hold harmful and harmless activations, shapes `(n, d)` and `(m, d)`, with different averages. Score each prompt by its dot product with the unit vector pointing from the harmless average to the harmful average. Return the fraction of (harmful, harmless) pairs in which the harmful prompt scores strictly higher, as a Python float.

### q50134
At one layer, the harmful and harmless averages come out exactly equal. What should a pipeline that builds a unit direction for every layer do with that layer?

### q50135
Someone hands you a vector and says it is the refusal direction. You have the harmful and harmless activations. Which check best tests the claim that it separates the two classes?

## Misconceptions

- **Average over the other axis.** Averaging along dimension 1 mixes the numbers inside one prompt and leaves one number per prompt. The class average averages along dimension 0.
- **The order of the subtraction does not matter.** Harmless minus harmful is the negative arrow. Harmful activations would then have negative scores along it.
- **The two classes need equal numbers of prompts.** Each class is averaged first, so a class of 4 and a class of 40 subtract fine.
- **Normalising changes the direction.** It rescales the vector to length 1 and leaves the direction alone.
- **A zero vector normalises to zeros.** Its length is 0, so dividing gives NaN. Handle that case on purpose.
