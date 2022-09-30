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

def divide_filters(size, width):
    if width > size:
        raise ValueError("Wider than size.")
    
    step = int(size / width)
    missing = size - (width * step)

    batches = [step] * width

    for idx in range(0, missing):
        batches[idx] += 1

    return batches

def ConvBlockBase(input_layer, size, residual=False, activation="relu", kernel_initializer="glorot_normal"):
    if residual:
        size = input_layer.shape[-1]

    conv1 = tf.keras.layers.Conv2D(size, 1, padding="same", activation=activation, kernel_initializer=kernel_initializer)(input_layer)
    conv1 = tf.keras.layers.Conv2D(size, 1, padding="same", activation=activation, kernel_initializer=kernel_initializer, use_bias=False)(conv1)

    conv2 = tf.keras.layers.Conv2D(size, 1, padding="same", activation=activation, kernel_initializer=kernel_initializer)(input_layer)
    conv2 = tf.keras.layers.Conv2D(size, 3, padding="same", activation=activation, kernel_initializer=kernel_initializer, use_bias=False)(conv2)

    conv3 = tf.keras.layers.Conv2D(size, 1, padding="same", activation=activation, kernel_initializer=kernel_initializer)(input_layer)
    conv3 = tf.keras.layers.Conv2D(size, 5, padding="same", activation=activation, kernel_initializer=kernel_initializer, use_bias=False)(conv3)

    merged = tf.keras.layers.Add()([conv1, conv2, conv3])
    merged = tf.keras.layers.BatchNormalization()(merged)
    merged = tf.keras.layers.Activation(activation)(merged)

    if residual:
        return tf.keras.layers.Add()([input_layer, merged])

    return merged

def ConvBlock(input_layer, size, depth=1, width=1, residual=False, activation="relu", kernel_initializer="glorot_normal"):
    
    wide_layers = []

    sizes = divide_filters(size, width)

    if residual:
        wide_layers.append(input_layer)

    for w in range(0, width):
        previous_depth = input_layer

        for d in range(0, depth):
            previous_depth = ConvBlockBase(previous_depth, sizes[w], residual=residual, activation=activation, kernel_initializer=kernel_initializer)

        wide_layers.append(previous_depth)

    if len(wide_layers) > 1:
        if residual:
            return tf.keras.layers.Add()(wide_layers)
        else:
            return tf.keras.layers.Concatenate()(wide_layers)
    
    return wide_layers[0]


def ReductionBlock(
    layer_input,
    activation="relu",
    kernel_initializer="glorot_normal",
):
    track1 = tf.keras.layers.Conv2D(layer_input.shape[-1], kernel_size=1, padding="same", strides=(1, 1), activation=activation, kernel_initializer=kernel_initializer)(layer_input)
    track1 = tf.keras.layers.Conv2D(layer_input.shape[-1], kernel_size=3, padding="same", strides=(2, 2), activation=activation, kernel_initializer=kernel_initializer, use_bias=False)(track1)
    track1 = tf.keras.layers.BatchNormalization()(track1)
    track1 = tf.keras.layers.Activation(activation)(track1)

    return track1

def ExpansionBlock(
    layer_input,
    activation="relu",
    kernel_size=3,
    kernel_initializer="glorot_normal",
):
    track1 = tf.keras.layers.Conv2D(layer_input.shape[-1], kernel_size=1, padding="same", strides=(1, 1), activation=activation, kernel_initializer=kernel_initializer)(layer_input)
    track1 = tf.keras.layers.Conv2DTranspose(layer_input.shape[-1], kernel_size=kernel_size, strides=(2, 2), activation=activation, padding="same", kernel_initializer=kernel_initializer, use_bias=False)(track1)
    track1 = tf.keras.layers.BatchNormalization()(track1)
    track1 = tf.keras.layers.Activation(activation)(track1)

    return track1

# Must be function, not object metric
def wrap_metric_ignoring_conf(metric, name):
    def metric_func(true, pred_and_conf):
        pred, _conf = tf.split(pred_and_conf, 2, axis=-1)

        return metric(true, pred)

    metric_func.__name__ = name

    return metric_func


def wrap_metric_ignoring_pred(metric, name):
    def metric_func(_true, pred_and_conf):
        _pred, conf = tf.split(pred_and_conf, 2, axis=-1)

        return metric(conf)

    metric_func.__name__ = name

    return metric_func
