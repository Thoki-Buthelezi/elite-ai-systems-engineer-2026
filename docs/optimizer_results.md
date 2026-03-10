# Optimizer Comparison Experiments

## Objective

The goal of this experiment was to compare several gradient-based optimization
algorithms implemented from first principles. The optimizers evaluated were:

- Adam
- AdaGrad
- RMSProp
- Momentum

The task used to evaluate the optimizers was a simple linear regression problem
where the model must recover the parameter θ from data generated using the
relationship:

y = 3x

If optimization is successful, the learned parameter should converge to
approximately θ ≈ 3.

---

# Dataset

Synthetic data was generated randomly using the following procedure:

- 100 samples were created
- Each sample contained a value x sampled from a uniform range [-10, 10]
- The target value was computed as:

y = 3x

This produces a perfectly linear dataset where the optimal model parameter is known.

---

# Model

The model used was a simple linear function:

ŷ = θx

The loss function used during training was Mean Squared Error (MSE).

The gradient of the loss with respect to θ is:

∇θ = 2x(ŷ − y)

This gradient was computed manually in the training loop.

---

# Experiment 1 — Mini-batch Training

### Configuration

- Training examples: 100
- Batch size: 20
- Iterations: 20

Optimizer learning rates:

| Optimizer | Learning Rate |
|----------|---------------|
| Adam | 0.001 |
| AdaGrad | 0.1 |
| RMSProp | 0.001 |
| Momentum | 0.01 |

### Observation

Under these settings, **Momentum was the only optimizer that consistently moved
toward the correct parameter value**.

The other optimizers showed unstable behaviour and did not reliably converge
to θ ≈ 3 within the small number of iterations.

### Interpretation

This behaviour is likely due to two factors:

1. The number of training iterations was very small.
2. The chosen learning rates may not have been well suited for the problem.

Because of these constraints, the optimizers did not have sufficient time to
stabilize their updates.

---

# Experiment 2 — Full Batch Training

### Configuration

The experiment was repeated with the following changes:

- Full dataset used for each gradient update
- Iterations increased to 5000
- Adam learning rate increased to 0.01

| Optimizer | Learning Rate |
|----------|---------------|
| Adam | 0.01 |
| AdaGrad | 0.1 |
| RMSProp | 0.001 |
| Momentum | 0.01 |

### Observation

With these settings, **all optimizers converged toward the correct parameter
value θ ≈ 3**.

The training became more stable due to:

- more iterations
- less noisy gradient estimates (full batch updates)

### Interpretation

The results suggest that the inconsistent behaviour observed in the first
experiment was caused mainly by training configuration rather than by the
optimizers themselves.

When given sufficient training iterations and stable gradient estimates,
all tested optimizers were able to find the correct solution.

---

# Summary

| Optimizer | Mini-batch Training | Full-batch Training |
|----------|---------------------|---------------------|
| Adam | inconsistent | converged |
| AdaGrad | inconsistent | converged |
| RMSProp | inconsistent | converged |
| Momentum | converged | converged |

Momentum showed the most reliable behaviour in the short-run mini-batch
setting. However, when training was extended and full-batch gradients were
used, all optimizers successfully converged to the optimal parameter.

---## Optimization Animation

The animation below shows how the parameter θ evolves during
training when using the Adam optimizer.

![Optimization Animation](../animations/adam_lr_animation.gif)



# Next Steps

Further experiments will include:

- systematic comparison of learning rates
- convergence speed analysis
- visualization of optimization trajectories
- animated plots showing parameter updates during training