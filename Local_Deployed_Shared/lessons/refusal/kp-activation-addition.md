---
kc: refusal.activation-addition
title: Add the direction to make a harmless prompt look harmful
new_syntax: []
concepts: [add-scaled-direction, add-at-one-layer]
supporting: ['refusal.diff-in-means', 'refusal.direction-ablation', 'cnn.forward-hooks']
previews: []
faded: [1787, 1788, 1789, 1790]
guided: []
independent: [1791, 1792, 1793, 1794, 1795, 1796, 50146, 50147, 50148]
integrated: [1797, 1798, 1799, 50149, 50150]
---

## Concept: Adding the direction shifts every score by the same amount

Ablation removes the direction. The opposite test adds it: take an activation x from a harmless prompt and form

x' = x + c r,

where r is the difference-in-means vector (not rescaled: it keeps its own length) and c is a number you choose, called the *coefficient*. With c = 1 you add exactly the gap between the harmful and harmless averages.

What this does to the score along the unit direction r̂ = r / ‖r‖ is simple. The score of x' is the score of x plus c‖r‖: every activation moves up by the same amount, whatever it started at. That differs from ablation, which sends every score to 0 no matter where it started. In particular, after adding r to all harmless activations their *average* score becomes exactly the average score of the harmful activations, because r is exactly the gap between those two averages along the direction.

A large c pushes beyond the harmful side. The saved run records c = 1 and c = 2.

## Worked example

Use the four toy harmless rows and the 4-row difference vector from the first page. The unit direction is r̂. Predict the shift before running: every score should go up by the length of r, about 3.05.

```python
import torch as t
B = t.tensor([[0.3, 1.5, 2.5, 0.1],
              [-0.1, 0.3, 1.9, -0.1],
              [-0.2, 1.2, 1.5, -0.6],
              [-0.5, 0.8, 1.9, -0.7]])
r = t.tensor([3.025, 0.05, 0.025, 0.35])
u = r / r.norm()
before = B @ u
after = (B + r) @ u
print(before)
print(after)
# Hidden checks
shift = after - before
assert t.allclose(shift, r.norm().expand(4), atol=1e-4)
assert abs(after.mean().item() - 2.9158) < 1e-3
```

The four scores rise by the same 3.05. The average of the new scores, 2.92, equals the average score of the four harmful rows. With c = 2 the shift doubles:

```python
double = (B + 2 * r) @ u
print(double)
# Hidden checks
assert t.allclose(double - before, 2 * r.norm().expand(4), atol=1e-4)
```

## Faded practice

### q1787
`X` holds activations of harmless prompts, shape `(n, d)`, `r` is the difference-in-means vector, shape `(d,)`, and `c` is a number. Return `X` with `c` times `r` added to every row.

```python starter
import torch as t


def solve(X, r, c):
    return X + _____ * _____
```

```python solution
import torch as t


def solve(X, r, c):
    return X + c * r
```

### q1788
`X` holds activations of harmless prompts, shape `(n, d)`, `r` is the difference-in-means vector, shape `(d,)`, and `c` is a number. Return the score of every row along the direction `r` points in (a vector of length 1) AFTER adding `c` times `r` to every row, shape `(n,)`.

```python starter
import torch as t


def solve(X, r, c):
    u = r / r.norm()
    return (X + c * r) _____ u
```

```python solution
import torch as t


def solve(X, r, c):
    u = r / r.norm()
    return (X + c * r) @ u
```

## Concept: Add it at one layer, at every position

Where the paper ablates at every layer, it adds at only one: the layer chosen by the selection rule. Adding at every layer would add the direction again and again, because each layer's output is added into the running activation. So the intervention is a forward hook on that one layer's output that returns `output + c * r`.

It is applied at *every token position* of the prompt. The output of a layer has shape `(b, s, d)`, and adding `r` of shape `(d,)` broadcasts over the batch and position axes, so every position moves by the same vector. Adding at only the last position would be a different, weaker experiment.

As before, remove the hook afterwards so that later runs of the model are unaffected.

## Worked example

The same two-layer toy network as on the ablation page, in its own self-contained setup. Add r = (1, 0) after the first layer only. Predict the output: the first layer copies the input (0, 3), the hook makes it (1, 3), and the second layer adds the two coordinates into the first.

```python
import torch as t
net = t.nn.Sequential(t.nn.Linear(2, 2, bias=False),
                      t.nn.Linear(2, 2, bias=False))
net[0].weight.data = t.tensor([[1., 0.], [0., 1.]])
net[1].weight.data = t.tensor([[1., 1.], [0., 1.]])
r = t.tensor([1., 0.])
x = t.tensor([0., 3.])

def hook(layer, inputs, output):
    return output + r

plain = net(x)
h = net[0].register_forward_hook(hook)
added = net(x)
h.remove()
print(plain, added)
# Hidden checks
assert plain.tolist() == [3.0, 3.0]
assert added.tolist() == [4.0, 3.0]
assert net(x).tolist() == [3.0, 3.0]
```

Now a prompt with two token positions. The hook adds `r` at both positions at once:

```python
xs = t.tensor([[[0., 3.], [2., 1.]]])
h = net[0].register_forward_hook(hook)
out = net(xs)
h.remove()
print(out)
# Hidden checks
assert out.tolist() == [[[4.0, 3.0], [4.0, 1.0]]]
```

Both positions moved: the second layer's output changed by the same hook-added vector at each.

## Faded practice

### q1789
`model` is a stack of linear layers, `r` is the vector to add, `x` is an input vector, and the number `c` is 1. Return the output of `model` on `x` when `r` is added to the output of the FIRST layer only. Leave `model` exactly as you found it.

```python starter
import torch as t


def solve(model, r, x):
    c = 1
    def hook(layer, inputs, output):
        return output + c * r
    handle = model[0].register_forward_hook(_____)
    out = model(x)
    handle._____()
    return out
```

```python solution
import torch as t


def solve(model, r, x):
    c = 1
    def hook(layer, inputs, output):
        return output + c * r
    handle = model[0].register_forward_hook(hook)
    out = model(x)
    handle.remove()
    return out
```

### q1790
`model` is a stack of linear layers, `r` is the vector to add, `x` is an input (a vector or a batch of rows), `k` is a layer position and `c` a number. Return the output of `model` on `x` when `c` times `r` is added to the output of layer `k` only. Leave `model` as you found it.

```python starter
import torch as t


def solve(model, r, x, k, c):
    def hook(layer, inputs, output):
        return output + c * r
    handle = model[_____].register_forward_hook(hook)
    out = model(x)
    handle.remove()
    return out
```

```python solution
import torch as t


def solve(model, r, x, k, c):
    def hook(layer, inputs, output):
        return output + c * r
    handle = model[k].register_forward_hook(hook)
    out = model(x)
    handle.remove()
    return out
```

## Solo practice

### q1791
`X` holds harmless activations, shape `(n, d)`, `r` is a nonzero vector, and `c` is a number. Return how much each row's score along the direction of `r` (a vector of length 1) changes when `c` times `r` is added to it, shape `(n,)`.

### q1792
`H` holds harmful activations, shape `(n, d)`, `B` holds harmless activations, shape `(m, d)` (their averages differ). Add the harmful-minus-harmless difference of the averages to every harmless row, then return the average score of the shifted rows along the unit direction of that difference, as a Python float.

### q1793
`X` holds activations, shape `(n, d)`, and `r` is a nonzero vector. Return `X` after two steps in this order: first add `r` to every row, then remove from every row the component along the unit direction of `r`. Same shape as `X`.

### q1794
`model` is a stack of linear layers applied to inputs of shape `(b, s, 2)` (b prompts, s token positions), `r` is a vector of length 2 and `k` is a layer position. Return the output of `model` on `x` when `r` is added at EVERY token position of the output of layer `k`. Leave `model` as you found it.

### q1795
`X` holds harmless activations, shape `(n, d)`, `r` is a nonzero vector, `cs` is a one-dimensional tensor of coefficients in increasing order and `target` is a number. For each coefficient the average score of `X + c r` along the unit direction of `r` is the average score of `X` plus c times the length of `r`. Return the SMALLEST coefficient in `cs` whose average score is at least `target`, or -1.0 if none reaches it. Python float.

### q1796
`model` is a stack of two linear layers, `r` is the vector to add, `x` is an input and `c` is a number. Return the output of `model` on `x` with `c` times `r` added to the output of the SECOND layer only, then the same call with the addition at the first layer only, stacked as a tensor of shape `(2, d)` (row 0 second layer, row 1 first layer). Leave `model` as you found it.

### q50146
A coefficient times the difference vector is added to every harmless activation. What happens to their scores along the unit direction?

### q50147
The direction is added at one layer, not at every layer the way removal is. What is the reason?

### q50148
A run with coefficient 2 is compared with a run with coefficient 1, on the same harmless prompts. What does the idea behind the method lead you to expect for the share of prompts refused?

## Integrated practice

### q1797
`model` is a stack of two linear layers, `H` and `B` hold harmful and harmless activations at the output of the first layer (shapes `(n, 2)` and `(m, 2)`, with different averages), `x` is an input vector and `k` is a layer position. Build `r`, the harmful average minus the harmless average (not rescaled), and return the output of `model` on `x` with `r` added to the output of layer `k`. Leave `model` as you found it.

### q1798
`model` is a stack of two linear layers, `u` is a vector of length 1, `r` is a vector to add and `x` is an input vector. Return a tensor of shape `(2, d)`. Row 0: the output when the component along `u` is removed from the output of EVERY layer. Row 1: the output when `r` is added to the output of the first layer only. Leave `model` as you found it.

### q1799
`H` holds harmful activations, shape `(n, d)`, `B` harmless activations, shape `(m, d)` (their averages differ), and `c` is a number. Let `r` be the harmful average minus the harmless average and `u` its unit direction. Return the gap that is LEFT between the average score of the harmful rows and the average score of the harmless rows after `c` times `r` has been added to every harmless row. Python float, harmful minus harmless.

### q50149
A learner adds the unit direction instead of the difference vector. How does the result compare with adding the difference vector?

### q50150
Removing the direction weakens refusal on harmful prompts, and adding it creates refusal on harmless prompts. What does running both tests show that either one alone does not?

## Misconceptions

- **Adding r moves every score by 1.** It moves every score along r̂ by the length of r, times the coefficient.
- **Adding a direction sets the score, like ablation does.** Ablation sends the score to 0. Addition shifts it by a fixed amount, so a score that was already high ends up higher still.
- **Add at every layer, as ablation does.** Each layer adds into the running activation, so adding at every layer would pile the direction on. The paper adds at the one selected layer.
- **Only the last token needs the direction.** The experiment adds it at every position of the prompt, the same way ablation removes it at every position.
- **Rescale r to length 1 first.** The addition uses r itself, which is what makes c = 1 close the gap between the two class averages.
- **Adding then ablating leaves the addition in place.** Ablation after the addition removes the whole component, so the addition leaves no trace along the direction.
