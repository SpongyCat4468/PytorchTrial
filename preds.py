import matplotlib.pyplot as plt
def plot_predictions(train_input, train_output, test_input, test_output, predictions=None):
    plt.figure(figsize=(12, 7))
    plt.scatter(train_input.cpu(), train_output.cpu(), c="b", s=4, alpha=0.6, label="Training data")
    plt.scatter(test_input.cpu(), test_output.cpu(), c="g", s=4, alpha=0.6, label="Testing data")

    if predictions is not None:
        plt.scatter(test_input.cpu(), predictions.cpu(), c="r", s=4, alpha=0.8, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.title("Polynomial Regression with Random Train/Test Split")
    plt.xlabel("X (normalized)")
    plt.ylabel("Y (normalized)")
    plt.grid(True, alpha=0.3)
    plt.show()