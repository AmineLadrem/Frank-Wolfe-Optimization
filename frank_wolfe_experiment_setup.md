# Experiment Setup

## 1. Experimental Goal

The goal of the experiments is to compare the practical performance of selected Frank-Wolfe variants on constrained classification problems. The experiments focus on empirical risk minimization with logistic regression losses under an explicit feasible constraint set.

The comparison will evaluate the methods according to objective value decrease, Frank-Wolfe duality gap, classification accuracy, runtime, sparsity of the iterates, and stability across datasets. The purpose is therefore not only to identify which method obtains the lowest final objective value, but also to assess the trade-offs between optimization progress, computational cost, and sparse model structure.

## 2. Datasets

The main experiments use at least two datasets. The recommended experimental choice is A9a and IJCNN1, both of which are binary classification datasets commonly used in optimization and machine learning benchmarks.

### A9a

A9a is a binary classification dataset derived from census income data. The task is to predict whether an individual belongs to one of two income categories based on demographic and employment-related features. It is useful for evaluating constrained logistic regression because it contains sparse categorical features after encoding.

Preprocessing should include encoding labels as `{-1, +1}`, normalizing or standardizing the feature columns, and preserving sparse matrix representations when appropriate. The same train/test split must be used for all methods.

### IJCNN1

IJCNN1 is a binary classification dataset frequently used as a benchmark for classification algorithms. The task is to assign each input example to one of two classes based on numerical features. It provides a second binary classification setting with different feature characteristics from A9a, which helps evaluate stability across datasets.

Preprocessing should include encoding labels as `{-1, +1}`, normalizing or standardizing the features, and using consistent train/test splits across all compared Frank-Wolfe variants.

### MNIST

MNIST may be used as an optional extension for multi-class classification. If included, the binary logistic regression formulation should be extended to a multi-class loss such as softmax cross-entropy. Since the main experiments are designed for binary logistic regression, MNIST is not required for the primary comparison.

## 3. Optimization Problem

The experiments consider constrained empirical risk minimization problems of the form:

```math
\min_{w \in C} F(w)
=
\frac{1}{n}
\sum_{i=1}^{n}
\ell(y_i, x_i^\top w)
+
\lambda r(w)
```

Here, `w \in \mathbb{R}^d` is the model parameter vector, `C` is the feasible constraint set, `\ell` is the classification loss, `r(w)` is an optional regularizer, and `\lambda \geq 0` is the regularization weight.

This formulation is suitable for Frank-Wolfe methods because the algorithms require a linear minimization oracle over `C` rather than Euclidean projection onto `C`.

## 4. Loss Functions

### Logistic Regression Loss

For binary labels `y_i \in {-1, +1}`, the primary loss is the logistic regression loss:

```math
\ell(y_i, x_i^\top w)
=
\log\left(1 + \exp(-y_i x_i^\top w)\right)
```

This loss is convex in `w`, making it appropriate for evaluating the standard convergence behavior of Frank-Wolfe methods on constrained convex optimization problems.

### Logistic Regression with Non-Convex Regularizer

A second objective uses the same logistic loss together with a non-convex regularizer:

```math
F(w)
=
\frac{1}{n}
\sum_{i=1}^{n}
\log\left(1 + \exp(-y_i x_i^\top w)\right)
+
\lambda r(w)
```

This objective may be non-convex because of the regularization term `r(w)`. The exact formula for the non-convex regularizer should be taken from page 9 of the referenced paper. No additional regularizer form is assumed here.

## 5. Constraint Set

The main experiments use the `\ell_1` ball as the primary feasible set:

```math
C = \{w \in \mathbb{R}^d : \|w\|_1 \leq \tau\}
```

where `\tau > 0` is a tunable radius.

The `\ell_1` ball is appropriate for these experiments because it encourages sparse solutions, has an efficient linear minimization oracle, avoids expensive Euclidean projection steps, and is common in Frank-Wolfe experiments. This constraint set is particularly useful for high-dimensional classification problems where sparse model parameters are desirable.

## 6. Frank-Wolfe Variants

The experiments compare two to three Frank-Wolfe variants. The proposed comparison includes the following three methods.

### Vanilla Frank-Wolfe

Vanilla Frank-Wolfe computes a linear minimization oracle solution at each iteration and updates the current iterate toward that solution. It is included as the baseline method because it is the simplest and most standard Frank-Wolfe algorithm.

### Away-Step Frank-Wolfe

Away-Step Frank-Wolfe augments the standard Frank-Wolfe direction with away directions that move away from previously selected atoms in the active set. This variant is included because it can reduce the weight assigned to suboptimal atoms and may improve convergence behavior on polytopal constraint sets such as the `\ell_1` ball.

### Pairwise Frank-Wolfe

Pairwise Frank-Wolfe transfers weight directly from an away atom to a Frank-Wolfe atom. It is included because it can produce more aggressive active-set updates and is often effective when sparse representations are important.

## 7. Linear Minimization Oracle

For the `\ell_1` ball, the linear minimization oracle at iteration `k` is:

```math
s_k
=
\arg\min_{\|s\|_1 \leq \tau}
\nabla F(w_k)^\top s
```

The solution is:

```math
s_k = -\tau \cdot \operatorname{sign}([\nabla F(w_k)]_j)e_j
```

where:

```math
j = \arg\max_i |[\nabla F(w_k)]_i|
```

Here, `e_j` denotes the `j`-th standard basis vector. This LMO is efficient because it only requires identifying the gradient coordinate with largest absolute value. It also makes the update sparse, since each LMO solution is a signed coordinate vector.

## 8. Step-Size Strategies

The experiments should compare at least two step-size strategies.

### Predefined Step Size

The first strategy uses the classical predefined Frank-Wolfe step size:

```math
\gamma_k = \frac{2}{k+2}
```

This rule is simple, deterministic, and does not require additional function evaluations.

### Line Search

The second strategy chooses the step size by minimizing the objective along the update direction:

```math
\gamma_k
=
\arg\min_{\gamma \in [0,1]}
F((1-\gamma)w_k + \gamma s_k)
```

Line search is expected to perform better in practice because it adapts the step size to the local objective geometry. However, it may be more expensive because it requires additional objective evaluations or a one-dimensional optimization routine at each iteration.

## 9. Evaluation Metrics

The following metrics should be recorded for every method and dataset:

- Training objective value
- Test objective value
- Frank-Wolfe duality gap
- Classification accuracy
- Runtime per iteration
- Total runtime
- Number of non-zero coefficients in `w`
- Number of iterations required to reach tolerance

The Frank-Wolfe duality gap is used as a certificate of approximate optimality. For a current iterate `w_k`, the gap is:

```math
g(w_k)
=
\max_{s \in C}
\nabla F(w_k)^\top (w_k - s)
```

Equivalently, using the LMO solution `s_k`:

```math
g(w_k)
=
\nabla F(w_k)^\top (w_k - s_k)
```

## 10. Stopping Criteria

Each method should stop when at least one of the following conditions holds:

```math
g(w_k) \leq \epsilon
```

or

```math
k \geq k_{\max}
```

or when the relative improvement in objective value is smaller than a prescribed tolerance.

Typical values are:

```math
\epsilon = 10^{-4}
```

```math
k_{\max} = 1000
```

The same stopping criteria should be used for all Frank-Wolfe variants to ensure a fair comparison.

## 11. Implementation Details

The implementation should be written in Python. Recommended libraries are `numpy`, `scipy`, `scikit-learn`, `matplotlib`, and `pandas`.

Features should be normalized or standardized before optimization. For binary logistic regression, labels should be encoded as `{-1, +1}`. Sparse matrix representations should be used when appropriate, especially for datasets such as A9a. Random seeds should be fixed for all randomized operations, including train/test splitting and any initialization choices.

All methods should use the same initialization. A natural choice is `w_0 = 0`, which is feasible for the `\ell_1` ball. Train/test splits, preprocessing steps, regularization weights, constraint radii, stopping tolerances, and maximum iteration counts should be identical across methods.

## 12. Reproducibility Protocol

The final experiments should be reproducible using a single Python notebook. The notebook should download or load the datasets, preprocess the data, run all selected Frank-Wolfe variants, save all recorded metrics, generate all figures, and use fixed random seeds throughout.

Each figure should have a number, a title, axis labels, a legend, and a short caption. Tables should clearly identify the dataset, method, step-size strategy, final objective value, final duality gap, accuracy, runtime, sparsity, and number of iterations.

All experimental settings should be reported explicitly, including dataset split sizes, normalization procedure, value of `\tau`, value of `\lambda`, stopping tolerance, maximum number of iterations, and relevant software environment details.