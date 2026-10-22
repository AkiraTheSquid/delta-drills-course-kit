---
kc: math.orthogonal-projection
kind: math
title: Components along a direction, and removing them
supporting: [math.dot-products-norms]
new_syntax: []
previews: []
concepts: [component-along-direction, remove-component]
faded: [50119, 50120]
guided: []
independent: [50121, 50122, 50123, 50124, 50125, 50126, 50127]
integrated: [50128, 50129, 50130]
---

## Concept: The component of x along a direction is one number

Pick a direction and write it as a unit vector $\hat r$, one whose length is $1$. How far does a vector $x$ reach in that direction? That is the dot product:
$$c = x\cdot\hat r .$$
The number $c$ is the component of $x$ along $\hat r$. It is positive when $x$ leans the same way as $\hat r$, negative when it leans against it, and zero when $x$ is perpendicular to $\hat r$. The part of $x$ that lies along $\hat r$ is that component times the direction:
$$c\,\hat r = (x\cdot\hat r)\,\hat r .$$
If the direction is given as a vector $r$ that is not unit length, divide by its length first, or use the equivalent form $\dfrac{x\cdot r}{r\cdot r}\,r$.

## Worked example

Take $\hat r=(1,0)$ and $x=(3,4)$. Pairing entries, $c=3\cdot1+4\cdot0=3$, and the part of $x$ along $\hat r$ is $3\hat r=(3,0)$. The other coordinate, $4$, is how far $x$ reaches perpendicular to $\hat r$, so it does not enter the component.

Now tilt the direction. With $\hat r=(3/5,\,4/5)$ (length $\sqrt{9/25+16/25}=1$), the same $x$ has $c=3\cdot\tfrac35+4\cdot\tfrac45=5$. Here $x$ lies exactly along $\hat r$, so the component equals the length of $x$.

## Faded practice

### q50119
r̂ = (1, 0) is a unit vector and x = (3, 4). What is the component of x along r̂, the number x · r̂?

## Concept: Removing the component leaves a perpendicular remainder

To delete the direction $\hat r$ from $x$, subtract the part of $x$ that lies along it:
$$x' = x - (x\cdot\hat r)\,\hat r .$$
Two facts follow. First, $x'$ has component $0$ along $\hat r$: the subtraction takes away exactly what the dot product measured. So $x'\cdot\hat r=0$, and the removed part and $x'$ are perpendicular. Second, because they are perpendicular, their squared lengths add: $\lVert x\rVert^2=c^2+\lVert x'\rVert^2$. Applying the removal a second time subtracts $0\cdot\hat r$ and changes nothing.

Only the direction matters for the removal, not the length of the vector that names it: $r$ and $2r$ point the same way, so they remove the same part. This is why the formula divides by the length, and why a direction is usually normalised before use.

## Worked example

Remove $\hat r=(0,1)$ from $x=(3,4)$. The component is $4$, the removed part is $(0,4)$, and what is left is $(3,0)$. Check: $(3,0)\cdot(0,1)=0$, and the squared lengths add up: $25=16+9$.

A tilted check: with $\hat r=(3/5,4/5)$ and $x=(5,0)$ the component is $3$, the removed part is $(9/5,12/5)$ and the remainder is $(16/5,-12/5)$, whose dot product with $\hat r$ is $\tfrac{48}{25}-\tfrac{48}{25}=0$.

## Faded practice

### q50120
x = (3, 4) and r̂ = (0, 1) is a unit vector. Remove the part of x along r̂, that is x − (x · r̂) r̂. What vector is left?

## Solo practice

### q50121
r̂ = (2/3, 1/3, 2/3) is a unit vector and x = (3, 6, 9). What is the component of x along r̂?

### q50122
r = (1, 1) is NOT a unit vector, and x = (4, 0). The part of x along r is ((x · r) / (r · r)) r. What is it?

### q50123
r̂ = (3/5, 4/5) is a unit vector and x = (5, 0). What is x with its part along r̂ removed?

### q50124
x has length 13 and its component along the unit vector r̂ is 5. After the part along r̂ is removed, how long is what is left?

### q50125
r̂ = (3/5, 4/5) is a unit vector and x = (5, 0). Remove the part of x along r̂, then measure the result's component along r̂ again. What is it?

### q50126
r̂ = (1, 0) is a unit vector and x = (−2, 1). Which statement about x and r̂ is correct?

### q50127
A teammate removes the part of an activation x along the unit direction r̂, and then, to be safe, removes the part along r̂ a second time. What does the second removal do?

## Integrated practice

### q50128
Harmful activations average (6, 2) and harmless activations average (2, 2). Take r as the harmful average minus the harmless average and r̂ as r divided by its length. What is x = (5, 3) after the part along r̂ is removed?

### q50129
x = (5, 3), r = (4, 0) and r̂ = (1, 0). First remove the part of x along r̂. Then add r to the result. What vector do you get?

### q50130
Two teams use the same activations. Team A calls the raw mean difference r and uses it directly. Team B uses r̂, the same vector divided by its length. Each team removes the part along its vector from x, and each adds its vector to x. Which comparison holds?
