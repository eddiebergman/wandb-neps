import wandb
import random
import time
import numpy as np

BATCH_SIZE_BIASES = {16: 1, 32: 2, 64: 1}


def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch, lr, bs):
    # NOTE(eddiebergman): This is a bias such that we bias better accuracy
    # depending on how far the learning_rate is from 0.001 (1e-3)
    # We also do the same based on batch size
    c = abs(-1 / (1 - (np.log(1e-3) - np.log(lr))))  # between (0, 1), 1 when it matches
    m = BATCH_SIZE_BIASES[bs]

    # Linear function w.r.t. epoch + noise, slop dictated by hyperparameters
    acc = 0.1 + ((epoch / 20) + (random.random() / 10)) * c * m
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main():
    run = wandb.init()

    # Note that we define values from `wandb.config`
    # instead of  defining hard values
    lr = wandb.config.lr
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs
    epoch_sleep_duration = wandb.config.epoch_sleep_duration

    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, bs)
        val_acc, val_loss = evaluate_one_epoch(epoch, lr, bs)

        data = {
            "epoch": epoch,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_loss": val_loss,
        }
        wandb.log(data, commit=True)
        print(data["val_acc"], epoch)
        time.sleep(epoch_sleep_duration)


# Call the main function.
if __name__ == "__main__":
    main()
