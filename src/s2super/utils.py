import os
import numpy as np
import tensorflow as tf
from glob import glob
from buteo.raster.patches import get_patches, get_kernel_weights, patches_to_array, weighted_median, mad_merge


def predict(
    model,
    data_input,
    data_output_proxy,
    number_of_offsets=9,
    tile_size=64,
    borders=True,
    batch_size=None,
    edge_distance=5,
    merge_method="mean",
):
    if isinstance(model, str):
        model = tf.keras.models.load_model(model) if isinstance(model, str) else model

    if not isinstance(data_input, list):
        data_input = [data_input]

    assert len(data_input) == len(model.inputs)

    overlaps = []
    for data in data_input:
        overlap, _, _ = get_patches(data, tile_size, number_of_offsets=number_of_offsets, border_check=borders)
        overlaps.append(overlap)

    _, offsets, shapes = get_patches(data_output_proxy, tile_size, number_of_offsets=number_of_offsets, border_check=borders)

    target_shape = list(data_output_proxy.shape)
    if len(target_shape) == 2:
        target_shape.append(len(offsets))
    else:
        target_shape[-1] = len(offsets)
    target_shape.append(1)

    arr = np.zeros(target_shape, dtype="float32")
    weights = np.zeros(target_shape, dtype="float32")
    weight_tile = get_kernel_weights(tile_size, edge_distance)

    model = tf.keras.models.load_model(model) if isinstance(model, str) else model

    for idx, offset in enumerate(offsets):
        og_shape = shapes[idx][0:2]; og_shape.append(1)
        og_shape = tuple(og_shape)

        test = []
        for overlap in overlaps:
            test.append(overlap[idx])

        predicted = model.predict(test, batch_size=batch_size)
        pred_reshaped = patches_to_array(predicted, og_shape, tile_size)
       
        pred_weights = np.tile(weight_tile, (predicted.shape[0], 1, 1))[:, :, :, np.newaxis]
        pred_weights_reshaped = patches_to_array(pred_weights, og_shape, tile_size)

        sx, ex, sy, ey = offset
        arr[sx:ex, sy:ey, idx] = pred_reshaped
        weights[sx:ex, sy:ey, idx] = pred_weights_reshaped

    weights_sum = np.sum(weights, axis=2)
    weights_norm = (weights[:, :, :, 0] / weights_sum)[:, :, :, np.newaxis]

    if merge_method == "mean":
        return np.average(arr, axis=2, weights=weights_norm)
    elif merge_method == "median":
        return weighted_median(arr, weights_norm)
    elif merge_method == "mad":
        return mad_merge(arr, weights_norm)

    return arr, weights_norm


def get_band_paths(safe_folder):
    bands = {
        "10m": {"B02": None, "B03": None, "B04": None, "B08": None, "AOT": None},
        "20m": {
            "B02": None,
            "B03": None,
            "B04": None,
            "B05": None,
            "B06": None,
            "B07": None,
            "B8A": None,
            "B11": None,
            "B12": None,
            "SCL": None,
            "AOT": None,
        },
        "60m": {
            "B01": None,
            "B02": None,
            "B03": None,
            "B04": None,
            "B05": None,
            "B06": None,
            "B07": None,
            "B8A": None,
            "B09": None,
            "B11": None,
            "B12": None,
            "SCL": None,
        },
        "QI": {
            "CLDPRB_20m": None,
            "CLDPRB_60m": None,
        },
    }

    assert os.path.isdir(safe_folder), f"Could not find folder: {safe_folder}"

    bands["QI"]["CLDPRB_20m"] = glob(
        f"{safe_folder}/GRANULE/*/QI_DATA/MSK_CLDPRB_20m.jp2"
    )[0]
    bands["QI"]["CLDPRB_60m"] = glob(
        f"{safe_folder}/GRANULE/*/QI_DATA/MSK_CLDPRB_60m.jp2"
    )[0]

    bands_10m = glob(f"{safe_folder}/GRANULE/*/IMG_DATA/R10m/*_???_*.jp2")
    for band in bands_10m:
        basename = os.path.basename(band)
        band_name = basename.split("_")[2]
        if band_name == "B02":
            bands["10m"]["B02"] = band
        if band_name == "B03":
            bands["10m"]["B03"] = band
        if band_name == "B04":
            bands["10m"]["B04"] = band
        if band_name == "B08":
            bands["10m"]["B08"] = band
        if band_name == "AOT":
            bands["10m"]["AOT"] = band

    bands_20m = glob(f"{safe_folder}/GRANULE/*/IMG_DATA/R20m/*.jp2")
    for band in bands_20m:
        basename = os.path.basename(band)
        band_name = basename.split("_")[2]
        if band_name == "B02":
            bands["20m"]["B02"] = band
        if band_name == "B03":
            bands["20m"]["B03"] = band
        if band_name == "B04":
            bands["20m"]["B04"] = band
        if band_name == "B05":
            bands["20m"]["B05"] = band
        if band_name == "B06":
            bands["20m"]["B06"] = band
        if band_name == "B07":
            bands["20m"]["B07"] = band
        if band_name == "B8A":
            bands["20m"]["B8A"] = band
        if band_name == "B09":
            bands["20m"]["B09"] = band
        if band_name == "B11":
            bands["20m"]["B11"] = band
        if band_name == "B12":
            bands["20m"]["B12"] = band
        if band_name == "SCL":
            bands["20m"]["SCL"] = band
        if band_name == "AOT":
            bands["20m"]["AOT"] = band

    bands_60m = glob(f"{safe_folder}/GRANULE/*/IMG_DATA/R60m/*_???_*.jp2")
    for band in bands_60m:
        basename = os.path.basename(band)
        band_name = basename.split("_")[2]
        if band_name == "B01":
            bands["60m"]["B01"] = band
        if band_name == "B02":
            bands["60m"]["B02"] = band
        if band_name == "B03":
            bands["60m"]["B03"] = band
        if band_name == "B04":
            bands["60m"]["B04"] = band
        if band_name == "B05":
            bands["60m"]["B05"] = band
        if band_name == "B06":
            bands["60m"]["B06"] = band
        if band_name == "B07":
            bands["60m"]["B07"] = band
        if band_name == "B8A":
            bands["60m"]["B8A"] = band
        if band_name == "B09":
            bands["60m"]["B09"] = band
        if band_name == "B11":
            bands["60m"]["B11"] = band
        if band_name == "B12":
            bands["60m"]["B12"] = band
        if band_name == "SCL":
            bands["60m"]["SCL"] = band
        if band_name == "AOT":
            bands["60m"]["AOT"] = band

    for outer_key in bands:
        for inner_key in bands[outer_key]:
            current_band = bands[outer_key][inner_key]
            assert (
                current_band != None
            ), f"{outer_key} - {inner_key} was not found. Verify the folders. Was the decompression interrupted?"

    return bands