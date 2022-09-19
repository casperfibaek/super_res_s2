import tensorflow as tf

class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric="val_loss", this_max=False, initial_weights=None):
        self.save_best_metric = save_best_metric
        self.max = this_max

        if initial_weights is not None:
            self.best_weights = initial_weights

        if this_max:
            self.best = float("-inf")
        else:
            self.best = float("inf")

    def on_epoch_end(self, _epoch, logs=None):
        metric_value = abs(logs[self.save_best_metric])
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()


class OverfitProtection(tf.keras.callbacks.Callback):
    def __init__(self, difference=0.1, patience=3, offset_start=3, verbose=True):
        self.difference = difference
        self.patience = patience
        self.offset_start = offset_start
        self.verbose = verbose
        self.count = 0

    def on_epoch_end(self, epoch, logs=None):
        loss = logs['loss']
        val_loss = logs['val_loss']
        
        if epoch < self.offset_start:
            return

        epsilon = 1e-7
        ratio = loss / (val_loss + epsilon)

        if (1.0 - ratio) > self.difference:
            self.count += 1

            if self.verbose:
                print(f"Overfitting.. Patience: {self.count}/{self.patience}")

        elif self.count != 0:
            self.count -= 1
        
        if self.count >= self.patience:
            self.model.stop_training = True

            if self.verbose:
                print(f"Training stopped to prevent overfitting. Difference: {ratio}, Patience: {self.count}/{self.patience}")
