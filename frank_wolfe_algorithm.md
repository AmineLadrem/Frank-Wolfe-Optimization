# Frank-Wolfe Algorithm

## Motivation

We want to optimize a constrained problem:

```math
\arg\min_{x \in C} f(x)
```

One possible approach is to take a gradient descent step and then project back onto the constraint set `C`.

However, this projection may not be cheap to compute.

The idea of Frank-Wolfe is to solve a linear minimization problem over `C` at each step, removing the need for a projection altogether. This is called the **Linear Minimization Oracle (LMO)**:

```math
s_k = \arg\min_{s \in C} \nabla f(x_k)^\top s
```

We then interpolate between our current position `x_k` and the result of the LMO `s_k` to get a new position:

```math
x_k = (1 - \gamma_k)x_{k-1} + \gamma_k s_{k-1}
```

In effect, instead of asking:

> How do we get back into the constrained set after stepping out?

Frank-Wolfe asks:

> Which point in `C` should I step towards?

We never violate the constraints at all.

---

## Definition

For a smooth, convex function `f` and a convex set `C`, we first calculate the **Linear Minimization Oracle (LMO)**:

```math
s_k = \arg\min_{s \in C} \nabla f(x_{k-1})^\top s
```

Then, the Frank-Wolfe update step is defined as:

```math
x_k = (1 - \gamma_k)x_{k-1} + \gamma_k s_{k-1}
```

where `\gamma_k` is the step size.

---

## Natural Language Explanation

At each step, `x_k` is somewhere inside or on the boundary of our constraint set `C`.

We then calculate the gradient:

```math
\nabla f(x_k)
```

at that point, which tells us the direction of steepest increase of `f`.

The LMO then asks:

> Which point in `C` would decrease `f` the most, based on the linear approximation of `f` at `x_k`?

We then linearly interpolate between our current position `x_k` and the result of the LMO `s_k` to derive a new position `x_{k+1}`.

The step size `\gamma_k` is the interpolation factor:

- At `\gamma = 0`, we do not move from our current position, so:

  ```math
  x_{k+1} = x_k
  ```

- At `\gamma = 1`, we immediately step fully to `s_k`, so:

  ```math
  x_{k+1} = s_k
  ```

---

# Optimality Condition

## Theorem

For a smooth, convex function `f` and a convex set `C`, the optimality condition is defined by the **duality gap**:

```math
f(x_k) - f(x^*) \leq \max_{s \in C} \left[\nabla f(x_k)^\top (x_k - s)\right]
```

We define:

```math
g(x_k) := \max_{s \in C} \left[\nabla f(x_k)^\top (x_k - s)\right]
```

such that:

```math
g(x_k) = 0 \iff f(x_k) = f(x^*)
```

---

## Proof

Let `x_k \in C` and let `x^* \in C` be the constrained minimizer for:

```math
\min_{x \in C} f(x)
```

We want to bound:

```math
f(x_k) - f(x^*)
```

That is, we want to measure how suboptimal we are, but we do not know `f(x^*)`.

---

### Step 1: Convexity

By assumption, `f` is convex. Therefore, we can lower bound `f(x^*)` by:

```math
f(x^*) \geq f(x_k) + \nabla f(x_k)^\top (x^* - x_k)
```

We can rearrange:

```math
f(x^*) \geq f(x_k) + \nabla f(x_k)^\top x^* - \nabla f(x_k)^\top x_k
```

---

### Step 2: Weaken the Lower Bound

Since `x^*` is the constrained minimizer, we know that:

```math
x^* \in C
```

Let `s_k` be the result of the LMO:

```math
s_k = \arg\min_{s \in C} \nabla f(x_k)^\top s
```

Since `x^* \in C` and `s_k` minimizes `\nabla f(x_k)^\top s` for all `s \in C`, we get:

```math
\min_{s \in C} \nabla f(x_k)^\top s \leq \nabla f(x_k)^\top x^*
```

Using this, we can weaken the right-hand side by substituting `\nabla f(x_k)^\top x^*` with:

```math
\min_{s \in C} \nabla f(x_k)^\top s
```

Thus:

```math
f(x^*) \geq f(x_k) + \min_{s \in C} \nabla f(x_k)^\top s - \nabla f(x_k)^\top x_k
```

---

### Step 3: Rearranging

We start with:

```math
f(x^*) \geq f(x_k) + \min_{s \in C} \nabla f(x_k)^\top s - \nabla f(x_k)^\top x_k
```

Move `f(x_k)` to the left:

```math
f(x^*) - f(x_k) \geq \min_{s \in C} \nabla f(x_k)^\top s - \nabla f(x_k)^\top x_k
```

Multiply both sides by `-1`:

```math
f(x_k) - f(x^*) \leq \nabla f(x_k)^\top x_k - \min_{s \in C} \nabla f(x_k)^\top s
```

Since `-\min[h] = \max[-h]`, we get:

```math
f(x_k) - f(x^*) \leq \nabla f(x_k)^\top x_k + \max_{s \in C} \left[-\nabla f(x_k)^\top s\right]
```

Since `\nabla f(x_k)^\top x_k` does not depend on `s`, we can pull it inside the maximum:

```math
f(x_k) - f(x^*) \leq \max_{s \in C} \left[\nabla f(x_k)^\top x_k - \nabla f(x_k)^\top s\right]
```

Therefore:

```math
f(x_k) - f(x^*) \leq \max_{s \in C} \left[\nabla f(x_k)^\top (x_k - s)\right]
```

So:

```math
f(x_k) - f(x^*) \leq g(x_k)
```

The quantity `g(x_k)` is the duality gap. It is an upper bound on the suboptimality:

```math
f(x_k) - f(x^*)
```

and can be computed at each iteration.

When `g(x_k)` is small, our current position `x_k` is close to `x^*`.

In particular:

```math
g(x_k) = 0 \Rightarrow f(x_k) = f(x^*)
```

---

## Inverse Direction

We can also show that:

```math
f(x_k) = f(x^*) \Rightarrow g(x_k) = 0
```

If `f(x_k) = f(x^*)`, then `x_k` is an optimal constrained point. This means that we cannot decrease `f` by moving in any direction in `C`.

```math
\nabla f(x^*)^\top (x^* - s) \leq 0, \quad \forall s \in C
```

Since this holds for all `s \in C`, taking the maximum over `C` gives:

```math
g(x^*) = \max_{s \in C} \left[\nabla f(x^*)^\top (x^* - s)\right] \leq 0
```

But since `x^* \in C`, we also get:

```math
\nabla f(x^*)^\top (x^* - x^*) = 0
```

Since `g(x^*)` maximizes over this quantity, we know that it is at most `0` and also contains a case where it is exactly `0`. Therefore:

```math
g(x^*) = 0
```

Thus:

```math
f(x_k) = f(x^*) \Rightarrow g(x_k) = 0
```

---

# Convergence

## Theorem

For an `L`-smooth, convex function `f` and a convex set `C`, the rate of convergence of Frank-Wolfe is:

```math
f(x_k) - f(x^*) \leq \frac{2M}{k + 2}
```

where `M` is the curvature constant:

```math
M =
\max_{\substack{
x, s \in C \\
y = (1-\gamma)x + \gamma s \\
\gamma \in [0,1]
}}
\frac{2}{\gamma^2}
\left(
f(y) - f(x) - \nabla f(x)^\top (y - x)
\right)
```

This gives us an `O(1/k)` convergence rate for Frank-Wolfe.

---

## Lemma

The curvature constant `M` is bounded by:

```math
M \leq L \cdot \operatorname{diam}^2(C)
```

---

## Proof

We start with the definition of `M`:

```math
M =
\max_{\substack{
x, s \in C \\
y = (1-\gamma)x + \gamma s \\
\gamma \in [0,1]
}}
\frac{2}{\gamma^2}
\left(
f(y) - f(x) - \nabla f(x)^\top (y - x)
\right)
```

From `L`-smoothness, we get:

```math
f(y) \leq f(x) + \nabla f(x)^\top (y - x) + \frac{L}{2}\|y - x\|^2
```

Rearranging gives:

```math
f(y) - f(x) - \nabla f(x)^\top (y - x)
\leq
\frac{L}{2}\|y - x\|^2
```

From the Frank-Wolfe update step, we know:

```math
x_{k+1} = (1 - \gamma_k)x_k + \gamma_k s_k
```

Now set:

```math
x_{k+1} = y, \quad x_k = x, \quad s_k = s, \quad \gamma_k = \gamma
```

Then:

```math
y = (1-\gamma)x + \gamma s
```

Rearranging:

```math
y = x - \gamma x + \gamma s
```

so:

```math
y - x = \gamma(s - x)
```

Plugging this into the right-hand side of the smoothness inequality gives:

```math
f(y) - f(x) - \nabla f(x)^\top (y - x)
\leq
\frac{L}{2}\|\gamma(s - x)\|^2
```

Simplifying:

```math
f(y) - f(x) - \nabla f(x)^\top (y - x)
\leq
\frac{L\gamma^2}{2}\|s - x\|^2
```

Next, multiply both sides by `2/\gamma^2`:

```math
\frac{2}{\gamma^2}
\left(
f(y) - f(x) - \nabla f(x)^\top (y - x)
\right)
\leq
L\|s - x\|^2
```

Now maximize both sides:

```math
\max
\frac{2}{\gamma^2}
\left(
f(y) - f(x) - \nabla f(x)^\top (y - x)
\right)
\leq
\max_{x,s \in C} L\|s - x\|^2
```

Using the definition of `M`, the left-hand side becomes:

```math
M \leq \max_{x,s \in C} L\|s - x\|^2
```

Since `L` does not depend on `x` or `s`, we can pull it out of the maximum:

```math
M \leq L \max_{x,s \in C} \|s - x\|^2
```

Since both `x \in C` and `s \in C`, the distance between them can be bounded by the size of `C`:

```math
\|s - x\| \leq \operatorname{diam}(C) =
\max_{a,b \in C}\|a - b\|
```

Therefore:

```math
M \leq L \cdot \operatorname{diam}^2(C)
```

which is the desired inequality.

---

## Natural Language Explanation of `M`

`M` measures how much `f` curves away from the linear approximation along Frank-Wolfe update directions `y`.

The more `f` bends, the less accurate the linear approximation is. This makes `M` larger and the convergence slower.

We use the measure `M` rather than `L` because `M` can be tighter. The constant `L` measures the maximum curvature in all directions, but Frank-Wolfe only ever moves along convex combinations within `C`. Therefore, `M` measures the curvature along those specific directions.

For example, imagine a three-dimensional space `\mathbb{R}^3`. Suppose that the function `f` curves only slightly along the `x` and `y` axes but curves strongly along the `z` axis.

Now assume that our constraint set `C` is only a plane within that space, along the `x` and `y` axes.

Since `L` captures the worst curvature in the whole space, but `M` only captures the worst curvature in convex combinations within `C`, we can see why:

```math
M \leq L \cdot \operatorname{diam}^2(C)
```

---

# One-Step Bound

## Proof

We start with the definition of the curvature constant:

```math
M =
\max_{\substack{
x, s \in C \\
y = (1-\gamma)x + \gamma s \\
\gamma \in [0,1]
}}
\frac{2}{\gamma^2}
\left(
f(y) - f(x) - \nabla f(x)^\top (y - x)
\right)
```

Since `M` is the maximum, it upper bounds any particular choice of `x`, `y`, and `\gamma`:

```math
M \geq
\frac{2}{\gamma^2}
\left(
f(y) - f(x) - \nabla f(x)^\top (y - x)
\right)
```

Rearranging:

```math
f(y) - f(x) - \nabla f(x)^\top (y - x)
\leq
\frac{\gamma^2}{2}M
```

Plug in:

```math
y = x_{k+1}, \quad x = x_k, \quad \gamma = \gamma_k
```

Then:

```math
f(x_{k+1}) - f(x_k) - \nabla f(x_k)^\top (x_{k+1} - x_k)
\leq
\frac{\gamma_k^2}{2}M
```

Rearranging to bound `f(x_{k+1})`:

```math
f(x_{k+1})
\leq
f(x_k) + \nabla f(x_k)^\top (x_{k+1} - x_k)
+
\frac{\gamma_k^2}{2}M
```

The Frank-Wolfe update step is:

```math
x_{k+1} = (1-\gamma_k)x_k + \gamma_k s_k
```

Rearranging:

```math
x_{k+1} = x_k + \gamma_k(s_k - x_k)
```

Therefore:

```math
x_{k+1} - x_k = \gamma_k(s_k - x_k)
```

Plugging this into the inequality gives:

```math
f(x_{k+1})
\leq
f(x_k) + \gamma_k \nabla f(x_k)^\top (s_k - x_k)
+
\frac{\gamma_k^2}{2}M
```

From the LMO, we know:

```math
s_k = \arg\min_{s \in C} \left[\nabla f(x_k)^\top s\right]
```

Minimizing a quantity is equivalent to maximizing its negation:

```math
s_k = \arg\max_{s \in C} \left[-\nabla f(x_k)^\top s\right]
```

Equivalently, since `x_k` does not depend on `s`:

```math
s_k = \arg\max_{s \in C} \left[\nabla f(x_k)^\top (x_k - s)\right]
```

From the duality gap, we know:

```math
g(x_k) = \max_{s \in C} \left[\nabla f(x_k)^\top (x_k - s)\right]
```

Since `s_k` maximizes this quantity:

```math
g(x_k) = \nabla f(x_k)^\top (x_k - s_k)
```

Negating both sides gives:

```math
-g(x_k) = \nabla f(x_k)^\top (s_k - x_k)
```

Now plug this into the one-step inequality:

```math
f(x_{k+1})
\leq
f(x_k) - \gamma_k g(x_k)
+
\frac{\gamma_k^2}{2}M
```

---

# Full Bound

Continuing with the previous inequality:

```math
f(x_{k+1})
\leq
f(x_k) - \gamma_k g(x_k)
+
\frac{\gamma_k^2}{2}M
```

From the optimality condition, we know:

```math
f(x_k) - f(x^*) \leq g(x_k)
```

Define:

```math
h(x_k) := f(x_k) - f(x^*)
```

So:

```math
h(x_k) \leq g(x_k)
```

Subtract `f(x^*)` from both sides of the one-step inequality:

```math
f(x_{k+1}) - f(x^*)
\leq
f(x_k) - f(x^*) - \gamma_k g(x_k)
+
\frac{\gamma_k^2}{2}M
```

Substitute in `h(x_k)`:

```math
h(x_{k+1})
\leq
h(x_k) - \gamma_k g(x_k)
+
\frac{\gamma_k^2}{2}M
```

Since:

```math
h(x_k) \leq g(x_k)
```

we have:

```math
-g(x_k) \leq -h(x_k)
```

Thus:

```math
h(x_{k+1})
\leq
h(x_k) - \gamma_k h(x_k)
+
\frac{\gamma_k^2}{2}M
```

Factorizing:

```math
h(x_{k+1})
\leq
(1-\gamma_k)h(x_k)
+
\frac{\gamma_k^2}{2}M
```

Choose the step size:

```math
\gamma_k = \frac{2}{k+2}
```

Plugging this in:

```math
h(x_{k+1})
\leq
\left(1 - \frac{2}{k+2}\right)h(x_k)
+
\frac{1}{2}
\left(\frac{2}{k+2}\right)^2 M
```

Simplifying:

```math
h(x_{k+1})
\leq
\frac{k}{k+2}h(x_k)
+
\frac{2M}{(k+2)^2}
```

---

## Induction Argument

We want to prove:

```math
h(x_k) \leq \frac{2M}{k+2}
```

by induction.

### Base Case

For `k = 0`, we need:

```math
h(x_0) \leq \frac{2M}{2} = M
```

We set this as a condition on the initialization `x_0`.

### Inductive Step

Assume:

```math
h(x_k) \leq \frac{2M}{k+2}
```

We want to show:

```math
h(x_{k+1}) \leq \frac{2M}{k+3}
```

Start with:

```math
h(x_{k+1})
\leq
\frac{k}{k+2}h(x_k)
+
\frac{2M}{(k+2)^2}
```

Plug in the induction hypothesis:

```math
h(x_{k+1})
\leq
\frac{k}{k+2}
\cdot
\frac{2M}{k+2}
+
\frac{2M}{(k+2)^2}
```

Simplify:

```math
h(x_{k+1})
\leq
\frac{2kM}{(k+2)^2}
+
\frac{2M}{(k+2)^2}
```

Thus:

```math
h(x_{k+1})
\leq
\frac{2M(k+1)}{(k+2)^2}
```

We want to show that:

```math
\frac{2M(k+1)}{(k+2)^2}
\leq
\frac{2M}{k+3}
```

Divide both sides by `2M`:

```math
\frac{k+1}{(k+2)^2}
\leq
\frac{1}{k+3}
```

Multiply both sides by `(k+2)^2`:

```math
k+1 \leq \frac{(k+2)^2}{k+3}
```

Multiply both sides by `k+3`:

```math
(k+1)(k+3) \leq (k+2)^2
```

Expand both sides:

```math
k^2 + 4k + 3 \leq k^2 + 4k + 4
```

which reduces to:

```math
3 \leq 4
```

Therefore, the inequality holds.

Hence:

```math
h(x_k) \leq \frac{2M}{k+2}
```

holds for all `k`.

Finally, using the definition of `h(x_k)`, we get:

```math
f(x_k) - f(x^*) \leq \frac{2M}{k+2}
```

which is the desired convergence bound.

---

## Note

In the original derivation, the line involving the inequality direction should be handled carefully.

Since:

```math
h(x_k) \leq g(x_k)
```

multiplying by `-1` gives:

```math
-g(x_k) \leq -h(x_k)
```

This is the inequality needed to replace `-g(x_k)` by the weaker upper bound `-h(x_k)` in the recurrence.
