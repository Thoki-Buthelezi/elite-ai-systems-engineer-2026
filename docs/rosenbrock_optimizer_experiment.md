# Optimizer Comparison on the Rosenbrock Function

## Objective

The purpose of this experiment was to evaluate the performance of several gradient-based optimization algorithms on a non-convex benchmark function.

The function used for this experiment is the **Rosenbrock function**, which is commonly used in optimization research due to its narrow curved valley that makes convergence challenging for many optimizers.

The Rosenbrock function is defined as:

f(x, y) = (1 − x)² + 100(y − x²)²

The global minimum occurs at:

x = 1  
y = 1  

where:

f(x, y) = 0

---

# Experimental Setup

Four optimizers implemented from first principles were evaluated:

- Adam
- AdaGrad
- Momentum
- RMSProp

Each optimizer was applied to minimize the Rosenbrock function starting from a randomly initialized point within the range:

[-3, 3]

for both parameters x and y.

### Stopping Condition

The optimization process stopped when either:

- the gradient norm satisfied  
|| gradient_norm || < 1e-6


or

- the maximum number of iterations (5000) was reached.

---

# Results

The following outcomes were observed:

| Optimizer | Convergence | Final Behavior |
|-----------|------------|----------------|
| Adam | Converged | Approached the global minimum |
| Momentum | Converged | Efficient movement through the curved valley |
| RMSProp | Converged | Stable convergence |
| AdaGrad | Poor convergence | Updates slowed significantly |

Example final parameter values observed during the experiment:
Adam:
z ≈ 0.141
x ≈ 0.625
y ≈ 0.389

AdaGrad:
z ≈ 0.034
x ≈ 0.814
y ≈ 0.661

Momentum:
z ≈ 0.000148
x ≈ 0.988
y ≈ 0.975

RMSProp:
z ≈ 0.022
x ≈ 0.982
y ≈ 0.978



Momentum produced the closest approximation to the true minimum.

---

# Visualization

The trajectories of each optimizer were plotted on top of the Rosenbrock contour surface.

This visualization shows how each optimizer navigates the curved valley of the loss surface.

![Optimizer Trajectories](../plots/rosenbrock_optimizer_paths.png)

---

# Discussion

The experiment highlights several important characteristics of the optimizers.

### Momentum
Momentum performed very well because it accumulates gradient direction over time.  
This helps reduce oscillation when moving through the narrow Rosenbrock valley.

### RMSProp
RMSProp adapts the step size using a running average of squared gradients.  
This produced stable convergence behavior.

### Adam
Adam combines momentum with adaptive learning rates, which typically leads to stable optimization.  
However, its performance in this experiment depended on the specific learning rate and initialization.

### AdaGrad
AdaGrad performed the worst in this experiment.  
Because AdaGrad accumulates squared gradients over time, its effective learning rate continuously decreases.  
As a result, parameter updates become extremely small and convergence slows significantly.

---

# Conclusion

This experiment demonstrates how different optimizers behave when minimizing a difficult non-convex function.

Momentum, RMSProp, and Adam successfully navigated the Rosenbrock valley and converged toward the global minimum.  
AdaGrad struggled due to its aggressively decreasing learning rate.

These results highlight how optimizer design influences convergence behavior on complex loss landscapes.

---

# Future Work

Possible extensions of this experiment include:

- comparing optimizer convergence speed
- running multiple trials with different initializations
- plotting loss vs iteration curves
- visualizing optimizer trajectories in 3D
