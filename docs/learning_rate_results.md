### Learning Rate Sensitivity (Adam)

A range of learning rates was evaluated to observe their effect on convergence.

Results showed that smaller learning rates resulted in slower convergence,
leading to higher final loss after a fixed number of training iterations.

Increasing the learning rate improved convergence speed within the tested
range. Adam remained stable even at relatively larger learning rates due
to its adaptive update rule.

Interestingly, the learning rate of 0.5 eventually achieved the lowest
final loss in this experiment. However, this behaviour is likely due to
the simplicity of the problem (single parameter linear regression).

In more complex neural network training scenarios, such a large learning
rate would typically lead to unstable training or divergence.