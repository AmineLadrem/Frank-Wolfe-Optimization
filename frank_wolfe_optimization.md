# Frank-Wolfe Optimization

The **Frank-Wolfe algorithm** is a projection-free optimization algorithm designed for constrained optimization problems. Unlike projected gradient methods, Frank-Wolfe avoids expensive projection steps by solving a **linear minimization oracle (LMO)** at each iteration.

This property makes it particularly attractive for large-scale optimization problems with structured constraints such as:

- Sparsity
- Low-rank matrices
- Simplex constraints

In recent years, Frank-Wolfe methods have gained renewed attention due to their scalability, interpretability, and applications in machine learning.

The purpose of this project is to survey and analyze Frank-Wolfe optimization methods from both theoretical and practical perspectives, focusing on:

- Convergence guarantees
- Algorithmic variants
- Applications in convex and non-convex optimization

## Objectives

- Conduct a comprehensive survey of the existing literature on the Frank-Wolfe algorithm.

- Study the mathematical foundations of the Frank-Wolfe method, including:

  - Linear minimization oracle (LMO)
  - Duality gap as a stopping criterion
  - Geometry of feasible sets and curvature constants

- Investigate different variants of Frank-Wolfe methods, such as:

  - Away-step Frank-Wolfe
  - Pairwise Frank-Wolfe
  - Fully corrective Frank-Wolfe
  - Stochastic and randomized Frank-Wolfe
  - Variance-reduced and accelerated variants

- Survey theoretical convergence guarantees under different assumptions, including:

  - Convex vs. strongly convex objectives
  - Non-convex extensions and their guarantees

- Identify the main advantages and limitations of Frank-Wolfe compared to projected gradient methods, including:

  - Sparsity of iterates
  - Scalability
  - Convergence speed

- Implement 2–3 selected Frank-Wolfe variants and evaluate their performance on two constrained optimization tasks and/or datasets.

## References

The following references are non-exhaustive:

- Lacoste-Julien, Simon, and Martin Jaggi. “On the global linear convergence of Frank-Wolfe optimization variants.” *Advances in Neural Information Processing Systems (NeurIPS)*, 2015.

- Jaggi, Martin. “Revisiting Frank-Wolfe: Projection-free sparse convex optimization.” *International Conference on Machine Learning (ICML)*, 2013.

# Experiments

## Datasets

You should run experiments on at least two of the following four datasets:

- **Covtype**: This is a dataset containing information about forest cover types. It is often used for classification problems.

- **A9a**: This is a dataset that is commonly used for binary classification and contains information about census records of people from the United States.

- **IJCNN1**: This is a benchmark dataset used in the field of machine learning and neural networks.

- **MNIST**: This is a dataset of handwritten digits that is often used for classification problems. Note that this is a multi-class problem.

## Loss Function

You should run your experiments using the following two losses:

- **Logistic regression loss**

  Logistic regression loss, also known as cross-entropy loss, is a popular classification loss used to model the probability of an outcome. It measures the difference between the predicted probability and the actual label.

  For binary classification problems, the formula for logistic regression loss is:

  ```math
  L(y, f(x)) = -[y \log(f(x)) + (1 - y)\log(1 - f(x))]
  ```

  where:

  - `y` is the true label, either `0` or `1`
  - `f(x)` is the predicted probability of the positive class
  - `log` is the natural logarithm

  The model `f(x)` should be a linear model:

  ```math
  f(x) = \langle w, x \rangle
  ```

  where `w` represents the model weights to be optimized.

- **Logistic regression with a non-convex regularizer**

  Logistic regression should also be tested with the non-convex regularizer `r(x)` defined in the following paper, page 9:

  <https://arxiv.org/pdf/1603.06159.pdf>

## Reproducibility and Source Code

Reproducibility is important in science because it allows for the independent verification and validation of research findings.

Therefore, you should deliver a Python notebook that can be easily run and that produces the figures presented in your report. These figures should be numbered and documented appropriately.
