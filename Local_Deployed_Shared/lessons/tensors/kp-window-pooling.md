---
kc: torch.window-pooling
title: Pooling with reshape and a reduction
supporting: [torch.reshape-flatten, torch.axis-reductions]
new_syntax: [Tensor.amax, Tensor.amax#dim]
concepts: [one-axis-windows, image-blocks]
faded: [1731, 1732, 1733, 1734]
guided: []
independent: [1735, 1736, 1739, 1737, 1738, 1740]
integrated: [1743, 1741, 1742]
---

## Concept: One axis: reshape into windows, reduce each row

Pooling shrinks an axis in two moves. First it cuts the axis into equal,
non-overlapping **windows** (chunks). Then it replaces every window with one
number: its mean, its max or its sum. Plain PyTorch does the first move with
`reshape` and the second with a reduction along one `dim`.

Start with six numbers:

```python
import torch as t

x = t.tensor([1., 2., 3., 4., 5., 6.])
print(x)
# Hidden checks
assert x.shape == (6,)
```

**Move 1, the split.** `reshape(3, 2)` reads the six numbers in order and
fills 3 rows of 2. Each row is one window, and neighbours in `x` share a
row: 1 and 2, then 3 and 4, then 5 and 6. Write the window size LAST and
let `-1` work out how many windows there are:

```python
import torch as t

x = t.tensor([1., 2., 3., 4., 5., 6.])
windows = x.reshape(-1, 2)    # -1 -> 6 / 2 = 3 rows
print(windows)
# Hidden checks
assert windows.tolist() == [[1., 2.], [3., 4.], [5., 6.]]
```

**Move 2, the collapse.** Each window is a row, so reduce along the row,
`dim=1`. The row axis disappears and one number per window is left:

```python
import torch as t

x = t.tensor([1., 2., 3., 4., 5., 6.])
pooled = x.reshape(-1, 2).mean(dim=1)
print(pooled)
# Hidden checks
assert pooled.tolist() == [1.5, 3.5, 5.5]
```

1.5 is the mean of 1 and 2, 3.5 of 3 and 4, 5.5 of 5 and 6.

**The window size is your choice.** The last reshape factor is the window;
the number of windows follows from it. Swap the reduction to change what
each window keeps:

```python
import torch as t

x = t.tensor([1., 2., 3., 4., 5., 6.])
print(x.reshape(-1, 3))              # 2 windows of 3
print(x.reshape(-1, 3).mean(dim=1))
print(x.reshape(-1, 2).amax(dim=1))  # the largest
# Hidden checks
assert x.reshape(-1, 3).mean(dim=1).tolist() == [2., 5.]
assert x.reshape(-1, 2).amax(dim=1).tolist() == [2., 4., 6.]
```

| window | `reshape` | the split | `.mean(dim=1)` |
|---|---|---|---|
| 1 | `(-1, 1)` | `[[1], [2], [3], [4], [5], [6]]` | `[1, 2, 3, 4, 5, 6]` |
| 2 | `(-1, 2)` | `[[1, 2], [3, 4], [5, 6]]` | `[1.5, 3.5, 5.5]` |
| 3 | `(-1, 3)` | `[[1, 2, 3], [4, 5, 6]]` | `[2, 5]` |
| 6 | `(-1, 6)` | `[[1, 2, 3, 4, 5, 6]]` | `[3.5]` |
| 4 | — | 6 is not divisible by 4 | error |

## Watch out

- **The window size goes LAST.** `x.reshape(2, -1)` also makes rows, but
  then reducing `dim=0` averages each COLUMN: `[1, 4]`, `[2, 5]`, `[3, 6]`,
  numbers 3 apart instead of neighbours. Reshape fills rows in reading
  order, so only the last factor holds neighbours.
- **Windows must tile the axis.** `x.reshape(-1, 4)` on 6 numbers raises
  "shape '[-1, 4]' is invalid for input of size 6". Real code pads first.
- **`mean` needs floats.** An integer tensor raises; `.to(t.float32)`
  first. `amax` and `sum` work on integers.

## Worked example

Twelve hourly readings; we want the total for each 4-hour shift. A shift is
a window of 4 hours, so 4 is the last reshape factor:

```python
import torch as t

hours = t.tensor([2., 1., 0., 3.,
                  5., 6., 4., 5.,
                  1., 1., 2., 0.])
shifts = hours.reshape(-1, 4)
print(shifts)
# Hidden checks
assert shifts.shape == (3, 4)
```

One row per shift, four hours per row. Now sum along the row:

```python
import torch as t

hours = t.tensor([2., 1., 0., 3.,
                  5., 6., 4., 5.,
                  1., 1., 2., 0.])
totals = hours.reshape(-1, 4).sum(dim=1)
print(totals)
# Hidden checks
assert totals.tolist() == [6.0, 20.0, 4.0]
```

Why each step:

1. `reshape(-1, 4)`: one row per shift. The `-1` means the same line works
   for a day or a week of readings.
2. `dim=1` is the axis of hours inside one shift, so it is the one that
   disappears.
3. `sum` instead of `mean` is the only difference from averaging. The
   windows stay the same.

## Faded practice

### q1731
Windows of `k` neighbours: the mean of each.

```python starter
import torch as t


def solve(x, k):
    """Mean of every window of k neighbours."""
    return x.reshape(_____).mean(dim=_____)
```

```python solution
import torch as t


def solve(x, k):
    """Mean of every window of k neighbours."""
    return x.reshape(-1, k).mean(dim=1)
```

### q1732
Daily readings, whole weeks: the largest reading of each week.

```python starter
import torch as t


def solve(temps):
    """Largest reading of each week."""
    return temps.reshape(_____)._____(dim=1)
```

```python solution
import torch as t


def solve(temps):
    """Largest reading of each week."""
    return temps.reshape(-1, 7).amax(dim=1)
```

## Concept: An image: split each axis in place, reduce two axes

An image has two axes, rows and columns, and 2 × 2 pooling cuts both. A
**block** is where a band of 2 rows crosses a band of 2 columns: a small
2 × 2 square of the image. Pooling collapses each block to one number.

```python
import torch as t

x = t.arange(16.).reshape(4, 4)
print(x)
# Hidden checks
assert x[1, 0] == 4.
```

**Step 1, split each axis where it stands.** Reshape can split an axis in
two, but it never moves one. So the height 4 becomes `(2, 2)` in place and
the width 4 becomes `(2, 2)` in place:

```python
import torch as t

x = t.arange(16.).reshape(4, 4)
H = x.shape[0]    # 4 rows
W = x.shape[1]    # 4 columns
blocks = x.reshape(H // 2, 2, W // 2, 2)
print(blocks.shape)
# Hidden checks
assert blocks.shape == (2, 2, 2, 2)
```

Read the four axes in order:

| axis | size | means |
|---|---|---|
| 0 | H/2 | which band of rows (which block, vertically) |
| 1 | 2 | which row INSIDE the block |
| 2 | W/2 | which band of columns (which block, sideways) |
| 3 | 2 | which column INSIDE the block |

So one block is `blocks[i, :, j, :]`: fix the two "which block" axes and
take everything inside:

```python
import torch as t

x = t.arange(16.).reshape(4, 4)
blocks = x.reshape(2, 2, 2, 2)
print(blocks[0, :, 0, :])   # top-left block
print(blocks[0, :, 1, :])   # the block to its right
# Hidden checks
assert blocks[0, :, 0, :].tolist() == [[0., 1.], [4., 5.]]
assert blocks[0, :, 1, :].tolist() == [[2., 3.], [6., 7.]]
```

**Step 2, collapse every block.** The inside of a block is axes 1 and 3,
so reduce exactly those two. Axes 0 and 2, the grid of blocks, are what is
left:

```python
import torch as t

x = t.arange(16.).reshape(4, 4)
H = x.shape[0]
W = x.shape[1]
blocks = x.reshape(H // 2, 2, W // 2, 2)
pooled = blocks.mean(dim=(1, 3))
print(pooled)
# Hidden checks
assert pooled.tolist() == [[2.5, 4.5], [10.5, 12.5]]
```

2.5 is the mean of 0, 1, 4 and 5, the top-left block. 12.5 is the mean of
10, 11, 14 and 15, the bottom-right one.

**Same shape, different axes.** For a 4 × 4 image, `reshape(2, 2, 2, 2)`
is ONE shape however you meant it. Reducing axes 2 and 3 instead of 1 and
3 averages a whole image row (four numbers in reading order), not a block:

```python
import torch as t

x = t.arange(16.).reshape(4, 4)
right = x.reshape(2, 2, 2, 2).mean(dim=(1, 3))
wrong = x.reshape(2, 2, 2, 2).mean(dim=(2, 3))
print(right)
print(wrong)   # 1.5 = mean of 0, 1, 2, 3
# Hidden checks
assert wrong.tolist() == [[1.5, 5.5], [9.5, 13.5]]
assert not t.equal(right, wrong)
```

**Batch and channel ride along.** Leave the leading axes as they are and
split only the last two. The in-block axes move to 3 and 5:

```python
import torch as t

batch = t.arange(32.).reshape(2, 1, 4, 4)
# (b, c, h, w) = (2, 1, 4, 4): split h and w only
out = batch.reshape(2, 1, 4 // 2, 2, 4 // 2, 2)
out = out.mean(dim=(3, 5))
print(out.shape)
# Hidden checks
assert out.shape == (2, 1, 2, 2)
```

## Watch out

- **Reduce the factors you inserted, not the last two axes.** After
  `(H // 2, 2, W // 2, 2)` the in-block axes are 1 and 3. With a batch and
  channel in front they are 3 and 5.
- **Blocks need not be square.** `reshape(H // 3, 3, W // 2, 2)` makes
  blocks of 3 rows by 2 columns, as long as 3 divides H and 2 divides W.
- **The first factor of each pair counts blocks, the second counts pixels
  in one block.** `(H // k, k)`, never `(k, H // k)`: the second would
  gather rows that are H/k apart.

## Worked example

A 6 × 4 heat map; we want the hottest value in each block of 3 rows by 2
columns. 6 rows make 2 bands of 3; 4 columns make 2 bands of 2. Look at one
block before reducing:

```python
import torch as t

heat = t.tensor([[1., 0., 2., 2.],
                 [0., 7., 1., 0.],
                 [3., 1., 0., 4.],
                 [0., 0., 5., 1.],
                 [2., 9., 0., 0.],
                 [1., 1., 3., 6.]])
blocks = heat.reshape(2, 3, 2, 2)
print(blocks[1, :, 0, :])   # rows 3-5, cols 0-1
# Hidden checks
assert blocks[1, :, 0, :].amax() == 9.
```

`blocks[1, :, 0, :]` is rows 3–5 and columns 0–1. Its largest value is 9.
Now the whole map, reducing the two in-block axes:

```python
import torch as t

heat = t.tensor([[1., 0., 2., 2.],
                 [0., 7., 1., 0.],
                 [3., 1., 0., 4.],
                 [0., 0., 5., 1.],
                 [2., 9., 0., 0.],
                 [1., 1., 3., 6.]])
peaks = heat.reshape(2, 3, 2, 2).amax(dim=(1, 3))
print(peaks)
# Hidden checks
assert peaks.tolist() == [[7.0, 4.0], [9.0, 6.0]]
```

Why each step:

1. `(2, 3, 2, 2)` is `(6 // 3, 3, 4 // 2, 2)`: each axis split in place,
   block count first, pixels-per-block second.
2. Check ONE block by hand (`blocks[1, :, 0, :]` → largest 9) and trust the
   reshape for the rest.
3. `amax(dim=(1, 3))` asks "how hot did it get anywhere in this block".

## Faded practice

### q1733
2×2 average pooling of one grayscale image.

```python starter
import torch as t


def solve(img):
    """2x2 average pooling."""
    h = img.shape[0]
    w = img.shape[1]
    return img.reshape(_____).mean(dim=_____)
```

```python solution
import torch as t


def solve(img):
    """2x2 average pooling."""
    h = img.shape[0]
    w = img.shape[1]
    return img.reshape(h // 2, 2, w // 2, 2).mean(dim=(1, 3))
```

### q1734
2×2 max pooling of every channel of a `(c, h, w)` image.

```python starter
import torch as t


def solve(img):
    """2x2 max pooling, every channel."""
    c = img.shape[0]
    h = img.shape[1]
    w = img.shape[2]
    return img.reshape(c, h // 2, 2, w // 2, 2).amax(dim=_____)
```

```python solution
import torch as t


def solve(img):
    """2x2 max pooling, every channel."""
    c = img.shape[0]
    h = img.shape[1]
    w = img.shape[2]
    return img.reshape(c, h // 2, 2, w // 2, 2).amax(dim=(2, 4))
```

## Solo practice

These drills practise the plain-PyTorch spelling, so an answer that calls
einops is marked wrong. The einops version is its own lesson.

### q1735
One axis, window `k`: the total of each window.

### q1736
A `(c, n)` signal: keep the channel axis, pool along time.

### q1739
Pool one axis only: average adjacent column pairs.

### q1737
k × k max pooling, any k.

### q1738
2×2 average pooling of a `(b, c, h, w)` batch.

### q1740
Blocks of `r` rows by `s` columns, summed.

## Integrated practice

### q1743
Two levels of windows at once: days, then 6-hour quarters inside each day.

### q1741
Pool, then flatten each batch item into one row.

### q1742
Subtract each block's own mean from its pixels: keep the reduced axes and
broadcast back.
