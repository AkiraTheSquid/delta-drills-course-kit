---
kc: einops.pooling
title: Pooling with factored axes
supporting: [einops.reduce-model, einops.split-axes, torch.window-pooling]
new_syntax: []
concepts: [one-axis-windows, image-blocks]
faded: [930, 929, 324, 363]
guided: []
independent: [931, 1641, 1639, 918, 917, 919, 1638, 1640, 368, 354]
integrated: [920, 336, 377]
---

## Concept: One axis: cut it into windows, collapse each window

Pooling shrinks an axis in two moves. First it cuts the axis into equal,
non-overlapping **windows** (chunks). Then it replaces every window with one
number: its mean, its max or its sum. einops writes both moves in one
pattern, and `rearrange` lets you look at the first move on its own.

Start with six numbers:

```python
import torch as t
import einops

x = t.tensor([1., 2., 3., 4., 5., 6.])
print(x)
# Hidden checks
assert x.shape == (6,)
```

**Move 1, the split.** Write the axis as `(h h2)`, which says "this axis of
6 is really h windows of h2 numbers each". You say how big one window is,
`h2=2`, and einops works out the other factor itself: h = 6 / 2 = 3.
`rearrange` keeps both names, so it shows the windows without computing
anything:

```python
import torch as t
import einops

x = t.tensor([1., 2., 3., 4., 5., 6.])
windows = einops.rearrange(x, '(h h2) -> h h2',
                           h2=2)
print(windows)
# Hidden checks
assert windows.tolist() == [[1., 2.], [3., 4.], [5., 6.]]
```

Every window is now one ROW. `h` counts the rows (3 windows). `h2`, the
second name inside the parentheses, is the length of each row, which is
the window size. Neighbours in `x` share a row: 1 and 2, then 3 and 4,
then 5 and 6.

**Move 2, the collapse.** `reduce` does the same split, but `h2` is missing
on the right of the arrow. A name that disappears is a name that gets
reduced, so each row of 2 collapses to its mean:

```python
import torch as t
import einops

x = t.tensor([1., 2., 3., 4., 5., 6.])
pooled = einops.reduce(x, '(h h2) -> h', 'mean',
                       h2=2)
print(pooled)
# Hidden checks
assert pooled.tolist() == [1.5, 3.5, 5.5]
```

1.5 is the mean of 1 and 2, 3.5 of 3 and 4, 5.5 of 5 and 6. Nothing else
happened: `reduce` is the `windows` grid from move 1, averaged along each
row. It is the same pooling as plain PyTorch's
`x.reshape(-1, 2).mean(dim=1)`, but the pattern names each factor, so
the window cannot end up in the wrong slot.

```python
import torch as t
import einops

x = t.tensor([1., 2., 3., 4., 5., 6.])
windows = einops.rearrange(x, '(h h2) -> h h2', h2=2)
pooled = einops.reduce(x, '(h h2) -> h', 'mean', h2=2)
print(windows.mean(dim=1))  # average each row
print(pooled)               # the one-call reduce
# Hidden checks
assert t.equal(windows.mean(dim=1), pooled)
```

**The window size is your choice.** `h2` picks the window; the number of
windows follows from it. The same six numbers with `h2=3` make 2 windows of
3:

```python
import torch as t
import einops

x = t.tensor([1., 2., 3., 4., 5., 6.])
print(einops.rearrange(x, '(h h2) -> h h2', h2=3))
print(einops.reduce(x, '(h h2) -> h', 'mean', h2=3))
# Hidden checks
assert einops.reduce(x, '(h h2) -> h', 'mean', h2=3).tolist() == [2., 5.]
```

Every window size that divides 6 works, and one that does not is an error:

| `h2` | windows `h` | the split | `'mean'` |
|---|---|---|---|
| 1 | 6 | `[[1], [2], [3], [4], [5], [6]]` | `[1, 2, 3, 4, 5, 6]` (nothing changes) |
| 2 | 3 | `[[1, 2], [3, 4], [5, 6]]` | `[1.5, 3.5, 5.5]` |
| 3 | 2 | `[[1, 2, 3], [4, 5, 6]]` | `[2, 5]` |
| 6 | 1 | `[[1, 2, 3, 4, 5, 6]]` | `[3.5]` |
| 4 | — | 6 is not divisible by 4 | error |

**The operation is the third argument.** The pattern holds the geometry,
the string holds what to do with each window. Same windows, keep the
largest:

```python
import torch as t
import einops

x = t.tensor([1., 2., 3., 4., 5., 6.])
print(einops.reduce(x, '(h h2) -> h', 'max', h2=2))
# Hidden checks
assert einops.reduce(x, '(h h2) -> h', 'max', h2=2).tolist() == [2., 4., 6.]
```

Step through it yourself. Change the window, change the operation, and
try a window size that does not divide the length:

```figure einops-pool-1d
Interactive figure: pick a window size h2, then step from the tensor, to the
rearranged windows (one row per window), to one number per window.
```

## Watch out

- **`(h h2)` and `(h2 h)` are different windows.** The window name has to be
  the SECOND (inner) one for neighbours to share a window. With `(h2 h)` and
  `h2=2`, the six numbers split as `[[1, 4], [2, 5], [3, 6]]`: each window
  takes numbers 3 apart. Same sizes, wrong windows. The figure's pattern
  switch shows it.
- **You must give the window size.** 6 can be 2 × 3 or 3 × 2, so
  `einops.reduce(x, '(h h2) -> h', 'mean')` stops with "Could not infer sizes".
  One keyword settles it; einops divides for the other.
- **Windows must tile the axis exactly.** `h2=4` on 6 numbers raises
  "can't divide axis of length 6 in chunks of 4". Real code pads first.

## Worked example

Twelve hourly readings; we want the total for each 4-hour shift. A shift is
a window of 4 hours, so the axis is `(shift hour)` with `hour=4`, and einops
infers 12 / 4 = 3 shifts. First look at the split:

```python
import torch as t
import einops

hours = t.tensor([2., 1., 0., 3.,
                  5., 6., 4., 5.,
                  1., 1., 2., 0.])
shifts = einops.rearrange(
    hours, '(shift hour) -> shift hour', hour=4)
print(shifts)
# Hidden checks
assert shifts.shape == (3, 4)
```

One row per shift, four hours per row. Now drop `hour` on the right and
sum it away:

```python
import torch as t
import einops

hours = t.tensor([2., 1., 0., 3.,
                  5., 6., 4., 5.,
                  1., 1., 2., 0.])
totals = einops.reduce(
    hours, '(shift hour) -> shift', 'sum', hour=4)
print(totals)
# Hidden checks
assert totals.tolist() == [6.0, 20.0, 4.0]
```

Why each step:

1. `hour` is the second name because the hours of one shift are
   neighbours. `(hour shift)` would put hours 0, 3, 6 and 9 in one "shift".
2. Only the window size is given. The number of shifts falls out of the
   length, so the same line works for a day or a week of readings.
3. `'sum'` instead of `'mean'` is the only difference from averaging. The
   windows stay the same.

## Faded practice

### q930
Readings in consecutive blocks of `k` days: the mean of each block.

```python starter
import torch as t
import einops


def solve(temps, k):
    """Block means, k days each."""
    x = t.tensor(temps)
    return einops.reduce(x, '_____', 'mean', _____=k).tolist()
```

```python solution
import torch as t
import einops


def solve(temps, k):
    """Block means, k days each."""
    x = t.tensor(temps)
    return einops.reduce(x, '(h k) -> h', 'mean', k=k).tolist()
```

### q929
Daily readings, whole weeks: the largest reading of each week.

```python starter
import torch as t
import einops


def solve(temps):
    """Weekly maximum."""
    x = t.tensor(temps)
    return einops.reduce(x, '_____', '_____', day=7).tolist()
```

```python solution
import torch as t
import einops


def solve(temps):
    """Weekly maximum."""
    x = t.tensor(temps)
    return einops.reduce(x, '(week day) -> week', 'max', day=7).tolist()
```

## Concept: An image: blocks of h2 rows by w2 columns

An image has two axes, rows and columns, and 2 × 2 pooling cuts both of
them. Each one gets its own split:

- `(h h2)` cuts the image's ROWS into h bands of h2 rows each.
- `(w w2)` cuts the image's COLUMNS into w bands of w2 columns each.

Where a band of rows crosses a band of columns there is a **block**: a
small h2 × w2 square of the image. Pooling collapses each block to one
number.

```python
import torch as t
import einops

x = t.arange(16.).reshape(4, 4)
print(x)
# Hidden checks
assert x[1, 0] == 4.
```

**Step 1, see the blocks.** Rearrange to `h w h2 w2`. The first two names
pick a block; the last two pick a number inside it.

```python
import torch as t
import einops

x = t.arange(16.).reshape(4, 4)
blocks = einops.rearrange(
    x, '(h h2) (w w2) -> h w h2 w2', h2=2, w2=2)
print(blocks.shape)
# Hidden checks
assert blocks.shape == (2, 2, 2, 2)
```

`(2, 2, 2, 2)`: a 2 × 2 grid of blocks, and every block is itself 2 × 2.
Print two of them:

```python
import torch as t
import einops

x = t.arange(16.).reshape(4, 4)
blocks = einops.rearrange(
    x, '(h h2) (w w2) -> h w h2 w2', h2=2, w2=2)
print(blocks[0, 0])
print(blocks[0, 1])
# Hidden checks
assert blocks[0, 0].tolist() == [[0., 1.], [4., 5.]]
assert blocks[0, 1].tolist() == [[2., 3.], [6., 7.]]
```

`blocks[0, 0]` is the top-left square, image rows 0–1 and columns 0–1:
`[[0, 1], [4, 5]]`. `blocks[0, 1]` is the square to its right,
`[[2, 3], [6, 7]]`. Inside a block, `h2` picks the row and `w2` the column,
so `blocks[0, 0, 1, 0]` is 4.

**Step 2, collapse every block.** Average over the two inner axes, 2 and 3.
Each 2 × 2 block becomes one number, and the grid of blocks is what is
left:

```python
import torch as t
import einops

x = t.arange(16.).reshape(4, 4)
blocks = einops.rearrange(
    x, '(h h2) (w w2) -> h w h2 w2', h2=2, w2=2)
print(blocks.mean(dim=(2, 3)))
# Hidden checks
assert blocks.mean(dim=(2, 3)).tolist() == [[2.5, 4.5], [10.5, 12.5]]
```

**Step 3, one call.** `reduce` does steps 1 and 2 together. It makes the
same split, and `h2` and `w2` are missing on the right, so they are the
names averaged away:

```python
import torch as t
import einops

x = t.arange(16.).reshape(4, 4)
blocks = einops.rearrange(
    x, '(h h2) (w w2) -> h w h2 w2', h2=2, w2=2)
pooled = einops.reduce(
    x, '(h h2) (w w2) -> h w', 'mean', h2=2, w2=2)
print(pooled)
# Hidden checks
assert t.equal(pooled, blocks.mean(dim=(2, 3)))
assert pooled.tolist() == [[2.5, 4.5], [10.5, 12.5]]
```

2.5 is the mean of 0, 1, 4 and 5, the top-left block. 12.5 is the mean of
10, 11, 14 and 15, the bottom-right one.

**Batch and channel ride along.** A name outside the parentheses is never
split, so it passes straight through. A `(b, c, H, W)` batch pools every
image and every channel on its own:

```python
import torch as t
import einops

batch = t.arange(32.).reshape(2, 1, 4, 4)
out = einops.reduce(
    batch, 'b c (h h2) (w w2) -> b c h w', 'mean',
    h2=2, w2=2)
print(out.shape)
# Hidden checks
assert out.shape == (2, 1, 2, 2)
```

Click through the blocks yourself. Try a 4 × 6 image, and blocks that are
not square:

```figure einops-pool-2d
Interactive figure: pick the block size (h2 rows by w2 columns), then step
from the image, to the pulled-apart blocks, to one number per block.
```

The same idea in the general console, with the loop it stands for —
including the windows before they are reduced:

```figure einops-console pooling
Interactive figure: step through '(h h2) -> h', the unreduced windows,
a 2 × 2 max pool and the blocks pulled apart.
```

## Watch out

- **Here `h2` counts rows of the IMAGE.** In the one-axis grid above, `h2`
  was the row length of the printed grid. In `(h h2) (w w2)`, `h2` is how
  many image rows sit in one block, and `w2` is how many image columns.
  Each axis is split on its own.
- **Blocks need not be square.** `h2=3, w2=2` is a valid 3-row by 2-column
  block, as long as 3 divides the height and 2 divides the width.
- **reduce cannot overlap windows.** The splits tile the image, so every
  number sits in exactly one block. Sliding (stride-1) pooling is a
  different tool.

## Worked example

A 6 × 4 heat map; we want the hottest value in each block of 3 rows by 2
columns. Blocks of 3 rows give h = 6 / 3 = 2 bands, blocks of 2 columns give
w = 4 / 2 = 2 bands. Look at one block before reducing:

```python
import torch as t
import einops

heat = t.tensor([[1., 0., 2., 2.],
                 [0., 7., 1., 0.],
                 [3., 1., 0., 4.],
                 [0., 0., 5., 1.],
                 [2., 9., 0., 0.],
                 [1., 1., 3., 6.]])
blocks = einops.rearrange(
    heat, '(h h2) (w w2) -> h w h2 w2', h2=3, w2=2)
print(blocks[1, 0])
# Hidden checks
assert blocks[1, 0].max() == 9.
```

`blocks[1, 0]` is rows 3–5 and columns 0–1. Its largest value is 9. Now
the whole map, with `'max'`:

```python
import torch as t
import einops

heat = t.tensor([[1., 0., 2., 2.],
                 [0., 7., 1., 0.],
                 [3., 1., 0., 4.],
                 [0., 0., 5., 1.],
                 [2., 9., 0., 0.],
                 [1., 1., 3., 6.]])
peaks = einops.reduce(
    heat, '(h h2) (w w2) -> h w', 'max', h2=3, w2=2)
print(peaks)
# Hidden checks
assert peaks.tolist() == [[7.0, 4.0], [9.0, 6.0]]
```

Why each step:

1. Check ONE block by hand (`blocks[1, 0]` → 9) and trust the pattern for
   the rest.
2. `h2=3` and `w2=2` are separate choices, one per axis. The block does not
   have to be square.
3. `'max'` asks "how hot did it get anywhere in this block", which is what
   max pooling keeps.

## Faded practice

### q324
2×2 non-overlapping average pooling on (B, C, H, W).

```python starter
import torch as t
import einops

def solve(x):
    """Halve H and W by averaging each 2x2 window."""
    return einops.reduce(x, '_____', 'mean', h2=2, w2=2)
```

```python solution
import torch as t
import einops

def solve(x):
    """Halve H and W by averaging each 2x2 window."""
    return einops.reduce(x, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=2, w2=2)
```

### q363
3×3 average pooling on one channels-first image (C, H, W), no batch axis.

```python starter
import torch as t
import einops

def solve(img):
    """3x3 mean pooling, channels unchanged."""
    return einops.reduce(img, '_____', 'mean', h2=3, w2=3)
```

```python solution
import torch as t
import einops

def solve(img):
    """3x3 mean pooling, channels unchanged."""
    return einops.reduce(img, 'c (h h2) (w w2) -> c h w', 'mean', h2=3, w2=3)
```

## Solo practice

Every drill from here is graded on its einops: the answer has to call
`einops.reduce`. A `reshape` and `.mean` give the same numbers, but they
practise a different lesson.

### q931
One axis, window 7: the total of each week.

### q1641
Temporal pooling with a sum: the window is the inner factor of time.

### q1639
Pool one axis only: a length-2 window on width alone.

### q918
2×2 mean pool, channels first.

### q917
2×2 max pool.

### q919
k × k min pool.

### q1638
Split each axis into (kept × 3), reduce the 3s away.

### q1640
Three factored axes: nothing new, one more group.

### q368
2×2×2 max pool of a 5-D volume batch.

### q354
Not a pooling: a reduce to kept singletons, then a broadcast subtraction.
Recognise the difference.

## Integrated practice

Each of these pools AND moves axes in the same reduce: the window factors
disappear while the kept axes are rearranged around them.

### q920
Pool, then side by side.

### q336
Max-pool AND tile the batch into a grid, in one reduce.

### q377
Max-pool, then flatten everything but the batch, in one reduce.
