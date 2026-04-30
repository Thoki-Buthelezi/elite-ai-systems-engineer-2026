# Optimization of the Rastrigin Function using Gradient-Based optimization

## Objective
The goal of this experiment was to evaluate the perfomance of 4 gradient-based optimization algorithms when minimising the **Rastrigin Function**, a well known non-convex benchmark function used in optimization research.

The optimizers evaluated inlclude:
-AdaGrad
-RMSProp
-Momentum
-Adam

The purpose of this experiment was to see how these optimizers behave when navigating a **highly multimodal search space**, where many local minima exists.

## Experimental Observations
The results from the visualization show that although some optimizers were able to locate **local minima**, none optimizer were able to escape those local minima to reach the global minima

This behaviour is expected due to highly multimodal nature of the Rastrigin function.

## RMSProp
RMSProp was able to converge to a local minimum. 
This is a result of RMSProp maintaining a moving weighted average of squared gradients. By normalize the gradient updates using accumulated squared average. this reduces oscillations and stabillizes updates

However, while RMSProp stabilizes convergence, it does not inherentlyy provide means to escape local minima.

## Adam
Adam also converged to a local minima. It combines momentum and RMSProp to stabilise learning. Similar to RMSProp, it also remained in local minimum.

## Momentum
Momentum was able to converge but did so slowly than RMSProp and Adam. Momemntum works by accumulating past gradients to smooth the update directions. This helps prevent oscillations and improves convergence stability.
Since the method follows a gradient, it also converged to a local minimum.

## AdaGrad
AdaGrad performed the worst among the 4 optimizers, AdaGrad adapts the learning rate by accumulating the sqaured of the previoius gradients. This makes the learning rate to premaature and excesively decrease

## Conclusion

This experiment demonstrates the behavior of several popular optimization algorithms on a highly non-convex function.

**Key findings include:**

Adam and RMSProp showed the most stable optimization behavior.

Momentum converged more slowly but still improved the optimization trajectory.

AdaGrad performed poorly due to the continual decay of its learning rate.

All optimizers became trapped in local minima, highlighting the limitations of gradient-based methods when dealing with highly multimodal landscapes.

The results suggest that escaping local minima may require more advanced optimization strategies such as:

-Simulated annealing
-Evolutionary algorithms
-Particle swarm optimization
-Random restarts