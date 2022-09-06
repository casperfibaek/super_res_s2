import os
import numpy as np
import tensorflow as tf
import cv2
from s2super.utils import predict, get_band_paths
from buteo.raster.patches import get_patches


def resample_array(arr, target_shape, interpolation=cv2.INTER_AREA):
    resized = cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=interpolation)

    return resized


def get_model():
    super_res_dir = os.path.dirname(os.path.realpath(__file__))
    model = tf.keras.models.load_model(os.path.join(super_res_dir, "SuperResSentinel_v4"))

    return model


def super_sample(
    data,
    fit_data=True,
    fit_epochs=5,
    indices={ "B02": 0, "B03": 1, "B04": 2, "B05": 4, "B06": 5, "B07": 6, "B08": 3, "B8A": 7, "B11": 8, "B12": 9 },
    method="fast",
    verbose=True,
    normalise=True,
    preloaded_model=None,
):
    """
    Super-sample a Sentinel 2 image. The source can either be a NumPy array of the bands, or a .safe file.

    ## Args:
    `data` (_str_/_np.ndarray_): The image to supersample </br>

    ## Kwargs:
    `indices` (_dict_): If the input is not a safe file, a dictionary with the band names and the indices in the NumPy array must be proved. It comes in the form of { "B02": 0, "B03": 1, ... } (Default: **10m first, then 20m**) </br>
    `method` (_str_): Either fast or accurate. If fast, uses less overlaps and weighted average merging. If accurate, uses more overlaps and the mad_merge algorithm (Default: **fast**) </br>
    `fit_data` (_bool_): Should the deep learning model be fitted with the data? Improves accuracy, but takes around 1m to fit on colab. (Default: **True**) </br>
    `fit_epochs` (_int_): If the model is refitted, for how many epochs should it run? (Default: **5**) </br>
    `verbose` (_bool_): If True, print statements will update on the progress (Default: **True**) </br>
    `normalise` (_bool_): If the input data should be normalised. Leave this True, unless it has already been done. The model expects sentinel 2 l2a data normalised by dividing by 10000.0 (Default: **True**) </br>
    `preloaded_model` (_None/tf.model_): Allows preloading the model, useful if applying the super_sampling within a loop. (Default: **None**) </br>

    ## Returns:
    (_np.ndarray_): A NumPy array with the supersampled data.
    """
    if isinstance(data, str):
        assert data.endswith(".SAFE")

        paths = get_band_paths(data)

        b10m = []
        for key in paths["10m"]:
            if verbose: print(f"Loading 10m band: {key}")
            if key in ["B02", "B03", "B04", "B08"]:
                img_10m = cv2.imread(paths["10m"][key], cv2.IMREAD_UNCHANGED)[:, :, np.newaxis]
                b10m.append(img_10m)

        for key in paths["20m"]:
            if key in ["B05", "B06", "B07", "B8A", "B11", "B12"]:
                if verbose: print(f"Loading 20m band: {key}")
                img_20m = cv2.imread(paths["20m"][key], cv2.IMREAD_UNCHANGED)
                if verbose: print(f"Resampling 20m band: {key}")
                img_20m = resample_array(img_20m, (b10m[0].shape[0], b10m[0].shape[1]), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]

                b10m.append(img_20m)
        
        data = np.concatenate(b10m, axis=2)

        b10m = None

    for band in ["B02", "B03", "B04"]:
        if band not in indices:
            assert "Bands 2, 3, and 4 are required to supersample other bands." 
    

    if preloaded_model is None:
        if verbose: print("Loading model...")
        super_res_dir = os.path.dirname(os.path.realpath(__file__))
        model = tf.keras.models.load_model(os.path.join(super_res_dir, "SuperResSentinel_v4"))
    else:
        model = preloaded_model

    if normalise:
        data = (data / 10000.0).astype("float32")

    if fit_data:
        if verbose: print("Re-fitting model...")
        for band in ["B02", "B03", "B04", "B08"]:
            if band not in indices:
                assert "Bands 2, 3, 4, and 8 are required to refit the model."

        rgb = data[:, :, [indices["B02"], indices["B03"], indices["B04"]]]
        y_train = data[:, :, indices["B08"]][:, :, np.newaxis]

        if verbose: print("Resampling target data.")
        nir_lr = resample_array(y_train, (y_train.shape[0] // 2, y_train.shape[1] // 2), interpolation=cv2.INTER_AREA)[:, :, np.newaxis]
        nir = resample_array(nir_lr, (y_train.shape[0], y_train.shape[1]), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]

        nir_patches, _, _ = get_patches(nir, tile_size=64, number_of_offsets=0, border_check=False)
        rgb_patches, _, _ = get_patches(rgb, tile_size=64, number_of_offsets=0, border_check=False)
        y_train, _, _ = get_patches(y_train, tile_size=64, number_of_offsets=0, border_check=False)
        x_train = [nir_patches, rgb_patches]

        lr = 0.00001
        model.optimizer.lr.assign(lr)

        if verbose: print("Fitting model...")
        model.fit(
            x=x_train,
            y=y_train,
            shuffle=True,
            epochs=fit_epochs,
            verbose=verbose,
            batch_size=32,
            use_multiprocessing=True,
            workers=0,
        )
    
    super_sampled = np.copy(data)

    for band in indices:
        if band in ["B02", "B03", "B04", "B08"]:
            super_sampled[:, :, indices[band]] = data[:, :, indices[band]]
        else:
            if verbose: print("Super-sampling band:", band)
            rgb = super_sampled[:, :, [indices["B02"], indices["B03"], indices["B04"]]]
            tar = super_sampled[:, :, indices[band]][:, :, np.newaxis]

            if method == "fast":
                pred = predict(model, [tar, rgb], tar, number_of_offsets=3, merge_method="mean", verbose=verbose)
            else:
                pred = predict(model, [tar, rgb], tar, number_of_offsets=9, merge_method="mad", verbose=verbose)

            super_sampled[:, :, indices[band]] = pred[:, :, 0]

    return np.rint(super_sampled * 10000.0).astype("uint16")
