---
kc: refusal.direction-ablation
title: Ablate the direction: remove its component everywhere
new_syntax: []
concepts: [direction-score, remove-component, ablate-every-layer]
supporting: ['refusal.diff-in-means', 'math.orthogonal-projection', 'torch.broadcasting-rules', 'torch.dot-matmul-patterns', 'cnn.forward-hooks']
previews: []
faded: [1772, 1773, 1774, 1775, 1776, 1777]
guided: []
independent: [1778, 1779, 1780, 1781, 1782, 1783, 50141, 50142, 50143]
integrated: [1784, 1785, 1786, 50144, 50145]
---

## Concept: Score each activation by how far it lies along the direction

You now have a unit direction u (length 1) for the refusal signal. The first question about any activation x is how much of it lies along u. That is one number, the dot product x · u, which this course calls the activation's *score*. A large positive score means x points the way harmful prompts point. A score near 0 means x is sideways to u.

For a table X with one activation per row, shape `(n, d)`, the dot products of all rows with u come out in one go as `X @ u`, shape `(n,)`. The same works for a batch of prompts with several token positions each, shape `(b, s, d)`: `X @ u` has shape `(b, s)`, one score per token, because `@` between a table and a vector sums over the last axis.

The score is not the first coordinate of x, and it is not the average of the coordinates. It uses every coordinate, weighted by u.

## Worked example

Take four activations from the toy run: the first and fourth harmful rows, then the first and fourth harmless rows. The direction is the unit direction from the previous page, rounded. Guess the sign pattern before you run it: the two harmful rows should score clearly higher than the harmless ones.

```python
import torch as t
u = t.tensor([0.9932, 0.0164, 0.0082, 0.1149])
X = t.tensor([[2.6, -0.5, 3.1, 0.0],
              [4.0, 1.3, 0.6, 0.6],
              [0.3, 1.5, 2.5, 0.1],
              [-0.5, 0.8, 1.9, -0.7]])
scores = X @ u
print(scores)
# Hidden checks
exp = t.tensor([2.5995, 4.0680, 0.3546, -0.5483])
assert t.allclose(scores, exp, atol=1e-3)
```

The harmful rows score 2.6 and 4.1, the harmless ones 0.35 and -0.55. The scores are positive and large for harmful prompts, and a score can be negative. Compare with just reading off the first coordinate of each row, which is a different thing:

```python
first = X[:, 0]
print(first)
# Hidden checks
assert t.allclose(first, t.tensor([2.6, 4.0, 0.3, -0.5]))
assert abs(first[1].item() - scores[1].item()) > 0.05
```

The first coordinates are close to the scores here only because u is nearly the first axis. Row 1 shows the gap: 4.0 against a score of 4.07.

## Faded practice

### q1772
`X` holds one activation per row, shape `(n, d)`, and `u` is a vector of length 1, shape `(d,)`. Return the dot product of every row with `u`, shape `(n,)`.

```python starter
import torch as t


def solve(X, u):
    return X _____ u
```

```python solution
import torch as t


def solve(X, u):
    return X @ u
```

### q1773
`X` holds activations for b prompts with s token positions each, shape `(b, s, d)`, and `u` is a vector of length 1, shape `(d,)`. Return, for each prompt, the average score of its tokens along `u`, shape `(b,)`.

```python starter
import torch as t


def solve(X, u):
    return (X _____ u).mean(dim=_____)
```

```python solution
import torch as t


def solve(X, u):
    return (X @ u).mean(dim=1)
```

## Concept: Remove the component along the direction

To *ablate* the direction from x means to remove exactly the part of x that lies along u and keep everything else. The part along u is the score times the direction, (x · u) u. Subtract it:

x' = x − (x · u) u.

After this, the new vector has score 0 along u: it has no part left along that direction, and it is otherwise unchanged. This is the projection you met on the mathematics page, applied to every row.

For a table X of shape `(n, d)` the scores have shape `(n,)`. To multiply each row's score by u, give the scores a trailing axis of length 1 with `[..., None]`, shape `(n, 1)`; then `scores[..., None] * u` broadcasts to `(n, d)`. Writing `X - X * u` is a different calculation: it multiplies coordinate by coordinate and does not use the dot product at all.

## Worked example

Remove the direction from the same four rows. The data is typed in again so this example stands alone. Predict the result of the check afterwards: the new scores along u should all be (almost) 0.

```python
import torch as t
u = t.tensor([0.9932, 0.0164, 0.0082, 0.1149])
X = t.tensor([[2.6, -0.5, 3.1, 0.0],
              [4.0, 1.3, 0.6, 0.6],
              [0.3, 1.5, 2.5, 0.1],
              [-0.5, 0.8, 1.9, -0.7]])
scores = X @ u
Y = X - scores[..., None] * u
print(Y)
# Hidden checks
assert Y.shape == X.shape
assert abs(Y[0, 2].item() - 3.0787) < 1e-3
```

Row 0 started as (2.6, -0.5, 3.1, 0.0). Now check that the direction is really gone from every row:

```python
print(Y @ u)
# Hidden checks
assert t.abs(Y @ u).max().item() < 1e-3
```

All four new scores are 0 (up to rounding in u). Row 0's first coordinate dropped from 2.6 to about 0.02, and its third barely moved (3.1 to 3.08), because u is almost entirely the first coordinate.

## Faded practice

### q1774
`X` holds one activation per row, shape `(n, d)`, and `u` is a vector of length 1, shape `(d,)`. Return `X` with the component along `u` removed from every row, same shape.

```python starter
import torch as t


def solve(X, u):
    return X - (X @ u)[..., _____] * _____
```

```python solution
import torch as t


def solve(X, u):
    return X - (X @ u)[..., None] * u
```

### q1775
`X` holds activations for b prompts with s token positions each, shape `(b, s, d)`, and `u` is a vector of length 1, shape `(d,)`. Return the part of every token that lies along `u` (the amount ablation would remove), same shape as `X`.

```python starter
import torch as t


def solve(X, u):
    return (X _____ u)[..., None] * _____
```

```python solution
import torch as t


def solve(X, u):
    return (X @ u)[..., None] * u
```

## Concept: Do it at every layer, with a forward hook

In the real model the activation is rewritten many times, layer by layer, and each layer *adds* its output into the running activation. If you remove the direction at just one place, a later layer can write it back in. So the paper removes the direction at every layer, from the first to the last, and at every token position.

In PyTorch, a *forward hook* lets you change a layer's output without editing the layer. You register a function with `layer.register_forward_hook(hook)`. The function receives the layer, its input and its output, and whatever it returns replaces the output. `register_forward_hook` returns a handle, and `handle.remove()` takes the hook off again. Always remove hooks when you are done, or the model keeps ablating every later run. In longer code, put the removal in the `finally:` part of a `try:` block, so the hook comes off even when the run raises an error.

The toy networks on this page are a short stack of linear layers, where each layer's output feeds the next. The real model also writes attention and feed-forward results back into a running sum, but the idea is the same: put the ablation on every layer's output.

## Worked example

A network of two linear layers, with the direction u = (1, 0) to remove. The first layer copies its input; the second adds the second coordinate into the first. Predict what happens if you ablate only after the first layer: the second layer can create a new first coordinate.

```python
import torch as t
net = t.nn.Sequential(t.nn.Linear(2, 2, bias=False),
                      t.nn.Linear(2, 2, bias=False))
net[0].weight.data = t.tensor([[1., 0.], [0., 1.]])
net[1].weight.data = t.tensor([[1., 1.], [0., 1.]])
u = t.tensor([1., 0.])
x = t.tensor([2., 3.])

def hook(layer, inputs, output):
    return output - (output @ u)[..., None] * u

plain = net(x)
h = net[0].register_forward_hook(hook)
first_only = net(x)
h.remove()
print(plain, first_only)
# Hidden checks
assert plain.tolist() == [5.0, 3.0]
assert first_only.tolist() == [3.0, 3.0]
assert net(x).tolist() == [5.0, 3.0]
```

Removing it after the first layer left a first coordinate of 3 in the output, because the second layer rebuilt it from the second coordinate. Now hook both layers:

```python
hs = [net[0].register_forward_hook(hook),
      net[1].register_forward_hook(hook)]
both = net(x)
for h in hs:
    h.remove()
print(both)
# Hidden checks
assert both.tolist() == [0.0, 3.0]
assert net(x).tolist() == [5.0, 3.0]
```

With the hook on every layer, the output has no first coordinate left, and removing the hooks restored the plain network.

## Faded practice

### q1776
`model` is a stack of linear layers, `u` is a vector of length 1 and `x` an input (a vector or a batch of rows). Return the output of `model` on `x` when the component along `u` is removed from the output of the FIRST layer only. Leave `model` exactly as you found it.

```python starter
import torch as t


def solve(model, u, x):
    def hook(layer, inputs, output):
        return output - (output @ u)[..., None] * u
    handle = model[0].register_forward_hook(_____)
    out = model(x)
    handle._____()
    return out
```

```python solution
import torch as t


def solve(model, u, x):
    def hook(layer, inputs, output):
        return output - (output @ u)[..., None] * u
    handle = model[0].register_forward_hook(hook)
    out = model(x)
    handle.remove()
    return out
```

### q1777
`model` is a stack of linear layers, `u` is a vector of length 1 and `x` an input. Return the output of `model` on `x` when the component along `u` is removed from the output of EVERY layer. Leave `model` exactly as you found it.

```python starter
import torch as t


def solve(model, u, x):
    def hook(layer, inputs, output):
        return output - (output @ u)[..., None] * u
    handles = []
    for layer in model:
        handles.append(layer.register_forward_hook(hook))
    out = model(x)
    for handle in _____:
        handle._____()
    return out
```

```python solution
import torch as t


def solve(model, u, x):
    def hook(layer, inputs, output):
        return output - (output @ u)[..., None] * u
    handles = []
    for layer in model:
        handles.append(layer.register_forward_hook(hook))
    out = model(x)
    for handle in handles:
        handle.remove()
    return out
```

## Solo practice

### q1778
`X` holds one activation per row, shape `(n, d)`, and `r` is a nonzero vector of any length, shape `(d,)`. Return the score of every row along the direction `r` points in, shape `(n,)`.

### q1779
`X` holds one activation per row, shape `(n, d)`, and `u` is a vector of length 1. Return True if no row has a component along `u` bigger in size than 0.0001 (so the direction is already removed), otherwise False.

### q1780
`X` holds activations for b prompts with s token positions each, shape `(b, s, d)`, and `r` is a nonzero vector of any length, shape `(d,)`. Return `X` with the component along the direction of `r` removed from every token, same shape.

### q1781
`model` is a stack of three linear layers, `u` has length 1, `x` is an input and `hooked` is a list of layer positions (0, 1 or 2). Return the output of `model` on `x` when the component along `u` is removed from the output of each layer whose position is in `hooked`, and from no other layer. Leave `model` as you found it.

### q1782
`model` is a stack of linear layers, `u` has length 1 and `x` is a single input vector. Remove the component along `u` from the output of the FIRST layer only, run the rest of the network, and return the score of the final output along `u`, as a Python float. Leave `model` as you found it.

### q1783
`model` is a stack of two linear layers, `u` has length 1 and `x` is an input. Return the output of `model` on `x` with the component along `u` removed from the output of every layer. The returned model must behave exactly as it did before your function ran.

### q50141
A unit direction is removed from a batch of prompts, harmful and harmless alike. Which statement is true of every prompt afterwards?

### q50142
The removal formula is x minus (the score of x) times the direction. Someone applies it with a direction that was never rescaled to length one. What goes wrong?

### q50143
Removing the direction at a single layer leaves the model refusing just as before. What best explains why the method removes it at every layer?

## Integrated practice

### q1784
`H` holds harmful activations, shape `(n, d)`, `B` harmless activations, shape `(m, d)` (the two class averages differ), and `X` any activations to clean, shape `(k, d)`. Build the direction that points from the harmless average to the harmful average, and return `X` with the component along that direction removed from every row.

### q1785
`H` and `B` hold harmful and harmless activations at L layers, shapes `(n, L, d)` and `(m, L, d)`, `l` is a layer position and `X` holds activations to clean at all L layers, shape `(b, L, d)`. Build the harmful-minus-harmless direction at layer `l`, rescale it to length 1, and remove its component from `X` at EVERY layer with that single direction. Return shape `(b, L, d)`.

### q1786
`H` holds harmful activations, shape `(n, d)`, not all zero, and `B` harmless activations, shape `(m, d)` (the class averages differ). Build the unit direction from the harmless average to the harmful average, and return the fraction of the harmful activations' total squared length that lies along it: the sum over rows of the squared score, divided by the sum over all entries of `H` squared. Python float between 0 and 1.

### q50144
The direction is built from activations at layer 18, yet the method also removes it at layer 3. What makes that a well-defined thing to do?

### q50145
Two engineers remove the direction at every layer. One measures only how often harmful prompts are still refused. The other also runs harmless prompts with the removal and compares the model's word probabilities with and without it. What does the second check add?

## Misconceptions

- **The score is the first coordinate, or the average of the coordinates.** It is the dot product with the unit direction, which weights every coordinate.
- **Subtract the direction itself.** `x - u` shifts every activation by the same amount. Ablation subtracts a different amount from each activation: its own score times u.
- **`X - X * u` is the projection.** That multiplies coordinate by coordinate. The projection needs the dot product first, then a multiple of u.
- **Any vector works as the direction.** The formula assumes length 1. A longer direction removes too much, a shorter one too little. Normalise first.
- **Ablating once is enough.** Later layers can write the direction back. The paper ablates at every layer and every token position.
- **Hooks can be left on.** A hook stays on the model after the call. Remove it, or the next run is silently ablated too.
