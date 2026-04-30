import json
import matplotlib.pyplot as plt

#open the file from result to read data from 
with open("optimizers/results/learning_rate_results.json", "r") as file:
    results = json.load(file)

plt.figure(figsize=(8,5))

for lr, data in results.items():
    loss_history = data["loss_history"]
    plt.plot(loss_history, label=f"lr={lr}")

plt.legend()
plt.xlabel("iterations")
plt.ylabel("loss")
plt.title("Adam Learning Rate Comparison")

plt.show()

plt.savefig("optimizers/plots/adam_learning_rates.py")