import matplotlib.pyplot as plt

def plot(history):

    keys = history.history.keys()

    if "acc" in keys:
        acc = history.history["acc"]
        val_acc = history.history["val_acc"]

        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, "bo", label="Training acc")
        plt.plot(epochs, val_acc, "b", label="Validation acc")
        plt.title("Training and validation accuracy")
        plt.legend()

        plt.savefig("./Images/Accuracy.png")

    if "loss" in keys:
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        epochs = range(1, len(loss) + 1)

        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()

        plt.savefig("./Images/Loss.png")
