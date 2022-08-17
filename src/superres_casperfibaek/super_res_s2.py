import os
import numpy as np
import tensorflow as tf
import cv2
from superres_casperfibaek.utils import get_overlaps, predict, get_band_paths


def resample_array(arr, target_shape, interpolation=cv2.INTER_AREA):
    resized = cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=interpolation)

    return resized


def super_sample_sentinel2(
    data,
    fit_data=True,
    indices={
        "B02": 0,
        "B03": 1,
        "B04": 2,
        "B05": 4,
        "B06": 5,
        "B07": 6,
        "B08": 3,
        "B8A": 7,
        "B11": 8,
        "B12": 9,
    },
    method="fast",
    verbose=True,
    normalise=True,
):
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
    
    if verbose: print("Loading model...")

    super_res_dir = os.path.dirname(os.path.realpath(__file__))
    model = tf.keras.models.load_model(os.path.join(super_res_dir, "SuperResSentinel_v2.h5"))

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

        nir_patches, _, _ = get_overlaps(nir, tile_size=64, number_of_offsets=0, border_check=False)
        rgb_patches, _, _ = get_overlaps(rgb, tile_size=64, number_of_offsets=0, border_check=False)
        y_train, _, _ = get_overlaps(y_train, tile_size=64, number_of_offsets=0, border_check=False)
        x_train = [nir_patches, rgb_patches]

        lr = 0.00001
        model.optimizer.lr.assign(lr)

        if verbose: print("Fitting model...")
        model.fit(
            x=x_train,
            y=y_train,
            shuffle=True,
            epochs=5,
            verbose=1,
            batch_size=32,
        )
    
    super_sampled = np.copy(data)

    for band in indices:
        if band in ["B02", "B03", "B04", "B08"]:
            super_sampled[:, :, indices[band]] = data[:, :, indices[band]]
        else:
            print("Super-sampling band:", band)
            rgb = super_sampled[:, :, [indices["B02"], indices["B03"], indices["B04"]]]
            tar = super_sampled[:, :, indices[band]][:, :, np.newaxis]

            if method == "fast":
                pred = predict(model, [tar, rgb], tar, number_of_offsets=3, merge_method="mean")
            else:
                pred = predict(model, [tar, rgb], tar, number_of_offsets=9, merge_method="mad")

            super_sampled[:, :, indices[band]] = pred[:, :, 0]

    return np.rint(super_sampled * 10000.0).astype("uint16")


s2_file = "/home/casper/Desktop/data/S2B_MSIL2A_20220804T101559_N0400_R065_T32TNR_20220804T130854.SAFE"
bob = super_sample_sentinel2(s2_file)
