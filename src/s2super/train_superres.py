import numpy as np
import tensorflow as tf
import numpy as np
from utils_train import SaveBestModel, OverfitProtection

MODEL_FOLDER = "./models/"
MODEL_NAME = "SuperResSentinel_v4"
TRAIN_DATASET = "./train_superres.npz"
TEST_SIZE = 0.2

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    print(e)

model_superres = tf.keras.models.load_model(MODEL_FOLDER + MODEL_NAME)

for layer in model_superres.layers:
    if "batch" in layer.name:
        layer.trainable = True

loaded = np.load(TRAIN_DATASET)
rgb = loaded["rgb"]
nir = loaded["nir"]
nir_lr = loaded["nir_lr"]

split_frac = int(rgb.shape[0] * 0.2)
rgb_train = rgb[:-split_frac]
nir_train = nir[:-split_frac]
nir_lr_train = nir_lr[:-split_frac]

rgb_test = rgb[-split_frac:]
nir_test = nir[-split_frac:]
nir_lr_test = nir_lr[-split_frac:]

def batch_generator(RGB, NIR, label, batchsize):
    random_shuffle = np.random.permutation(len(RGB))
    patches_len = len(RGB)
    idx = 0

    RGB_Shuffled = RGB[random_shuffle]
    NIR_Shuffled = NIR[random_shuffle]
    LABEL_Shuffled = label[random_shuffle]

    while True:
        yield [RGB_Shuffled[idx:idx + batchsize], NIR_Shuffled[idx:idx + batchsize]], LABEL_Shuffled[idx:idx + batchsize]
        idx = idx + batchsize

        if idx + batchsize > patches_len:
            idx = 0
            random_shuffle = np.random.permutation(len(RGB))
            RGB_Shuffled = RGB[random_shuffle]
            NIR_Shuffled = NIR[random_shuffle]
            LABEL_Shuffled = label[random_shuffle]

base_lr = 1e-4

fits = [
    { "epochs": 10, "bs": 128, "lr": base_lr},
    { "epochs": 10, "bs": 140, "lr": base_lr},
    { "epochs": 10, "bs": 156, "lr": base_lr},
    { "epochs": 10, "bs": 168, "lr": base_lr},
    { "epochs": 10, "bs": 180, "lr": base_lr},
]

cur_sum = 0
for nr, val in enumerate(fits):
    fits[nr]["ie"] = cur_sum
    cur_sum += fits[nr]["epochs"]

val_loss = model_superres.evaluate(
    batch_generator(nir_lr_test, rgb_test, nir_test, 512),
    batch_size=64,
    steps=int(nir_test.shape[0] / 512),
)

best_val_loss = val_loss
save_best_model = SaveBestModel(save_best_metric="val_loss", initial_weights=model_superres.get_weights())

for idx, fit in enumerate(range(len(fits))):
    use_epoch = fits[fit]["epochs"]
    use_bs = fits[fit]["bs"]
    use_lr = fits[fit]["lr"]
    use_ie = fits[fit]["ie"]

    model_superres.optimizer.lr.assign(use_lr)

    model_superres.fit(
        x=batch_generator(nir_lr_train, rgb_train, nir_train, use_bs),
        validation_data=batch_generator(nir_lr_test, rgb_test, nir_test, use_bs),
        shuffle=True,
        epochs=use_epoch + use_ie,
        initial_epoch=use_ie,
        batch_size=use_bs,
        steps_per_epoch=int(nir_train.shape[0] / use_bs),
        validation_steps=int(nir_test.shape[0] / use_bs),
        use_multiprocessing=True,
        workers=0,
        verbose=1,
        callbacks=[
            save_best_model,
            OverfitProtection(
                patience=5,
                difference=0.25, # 20% overfit allowed
                offset_start=3, # disregard overfit for the first epoch
            ),
        ],
    )

    model_superres.set_weights(save_best_model.best_weights)

    best_val_loss = model_superres.evaluate(
        batch_generator(nir_lr_test, rgb_test, nir_test, 512),
        batch_size=64,
        steps=int(nir_test.shape[0] / 512),
    )
    model_superres.save(f"{MODEL_FOLDER}SuperResSentinel_v5_{str(idx)}")

model_superres.save(f"{MODEL_FOLDER}SuperResSentinel_v5")
