# Week 05-06 Neural Network from Scratch

## Objective
- Understand forward propagation in vectorized form  
- Derive and implement backpropagation from scratch  
- Extend implementation to batch processing  

## 1. Foward Pass (Vectorized Form)
Each layer performs a linear transformation followed by a non-linearity.
Matrix multiplication allows computing all neurons simultaneously.
**Equations**
Z[l] = W[l]A[l-1] + b[l]  
A[l] = g(Z[l])

where the shapes are:
W[l] → (n[l], n[l-1])  
b[l] → (n[l], 1)  
A[l-1] → (n[l-1], m)  

## 2. Backpropagation (Vectorized Form)
Backpropagation computes gradients using the chain rule, propagating error signals backward through the network.
**Output Layer**
δ[2] = (A[2] - Y) ⊙ g'(Z[2])  
dW[2] = (1/m) δ[2] A[1]^T  
db[2] = (1/m) sum(δ[2])
**Hidden Layer**
δ[1] = (W[2]^T δ[2]) ⊙ g'(Z[1])  
dW[1] = (1/m) δ[1] X^T  
db[1] = (1/m) sum(δ[1])

## Batch Processing
Each column represents one training example.
Vectorization allows computing all examples simultaneously.

where the shapes are:
X → (n_x, m)  
A → (neurons, m)  

## 4. Key Lessons Learnt
- Gradients are always with respect to pre-activation (Z), not activation (A)  
- Weight gradients use matrix multiplication, not element-wise ops  
- Bias is one value per neuron, broadcast across examples  
- Shape consistency is the key to debugging  

## 5. Mistakes I Made
- Initially used sigmoid_derivative(A) instead of sigmoid_derivative(Z)  
- Used element-wise multiplication instead of matrix multiplication for dW  

## 6.Implementation
see: src/neural_networks/neural_net.py


