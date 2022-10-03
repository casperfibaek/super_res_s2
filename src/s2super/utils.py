import os
import numpy as np
import tensorflow as tf
from glob import glob
import buteo as beo


def predict(
    model,
    data_input,
    data_output_proxy,
    number_of_offsets=9,
    tile_size=64,
    borders=True,
    batch_size=256,
    edge_distance=3,
    merge_method="mean",
    merge_weights="both",
    output_confidence=False,
    output_variance=False,
    verbose=0,
):
    """ Create a prediction of sharpened sentinel 2 band.
    
    ## Args:
    `model` (_str_/_tf.model_): The S2Super model to use. Either path or loaded.\n
    `data_input` (_np.ndarray_): Input data for the model. Should be [bilinear_10m_unsharp, rgb_10m_sharp].\n
    `data_output_proxy` (_np.ndarray_): A reference to the shape of the output. Usually you can just use unsharp band.\n

    ## Kwargs:
    `number_of_offsets` (_int_): How many overlaps should be used for the prediction. (Default: **9**)\n
    `tile_size` (_int_): The square tilesize of the patches (Default: **64**)\n
    `borders` (_bool_): Should borders patches be added? True ensures the same output size as the input can be made. (Default: **True**)\n
    `batch_size` (_int_): Explicitely set the batch_size to use during inference. (Default: **256**)\n
    `preloaded_model` (_None/tf.model_): Allows preloading the model, useful if applying the super_sampling within a loop. (Default: **None**)\n
    `edge_distance` (_int_): Pixels closer to the edge will be weighted lower than central ones. What distance should be considered the maximum? (Default: **3**)\n
    `merge_method` (_str_): How should the predictions be merged? All methods are weighted. (max_conf, mean, median, mad) (Default: **mean**)\n
    `merge_weights` (_str_): How should the weights of the merge be calculated. (tile, conf, both, none) (Default: **both**)\n
    `output_confidence` (_bool_): Should the model output the confidence band, as well as the sharpened band? (Default: **False**)\n
    `output_variance` (_bool_): Should the model output the variance of the merged bands, as well as the sharpened band? (Default: **False**)\n
    `verbose` (_int_): Set the verbosity level of tensorflow. (Default: **1**)\n

    ## Returns:
    (_np.ndarray_): A NumPy array with the supersampled data.
    
    """
    if isinstance(model, str):
        model = tf.keras.models.load_model(model) if isinstance(model, str) else model

    if not isinstance(data_input, list):
        data_input = [data_input]

    assert len(data_input) == len(model.inputs)

    overlaps = []
    for data in data_input:
        overlap, _, _ = beo.get_patches(data, tile_size, number_of_offsets=number_of_offsets, border_check=borders)
        overlaps.append(overlap)

    _, offsets, shapes = beo.get_patches(data_output_proxy, tile_size, number_of_offsets=number_of_offsets, border_check=borders)

    target_shape = list(data_output_proxy.shape)
    if len(target_shape) == 2:
        target_shape.append(len(offsets))
    else:
        target_shape[-1] = len(offsets)
    target_shape.append(1)

    arr = np.zeros(target_shape, dtype="float32")
    weights = np.zeros(target_shape, dtype="float32")
    weight_tile = beo.get_kernel_weights(tile_size, edge_distance)

    model = tf.keras.models.load_model(model) if isinstance(model, str) else model

    for idx, offset in enumerate(offsets):
        og_shape = shapes[idx][0:2]; og_shape.append(1)
        og_shape = tuple(og_shape)

        test = []
        for overlap in overlaps:
            test.append(overlap[idx])

        predicted = model.predict(test, batch_size=batch_size, verbose=verbose)

        pred_values = predicted[:, :, :, 0][:, :, :, np.newaxis]
        conf_values = predicted[:, :, :, 1][:, :, :, np.newaxis]

        pred_reshaped = beo.patches_to_array(pred_values, og_shape, tile_size)
        conf_reshaped = beo.patches_to_array(conf_values, og_shape, tile_size)

        if merge_weights == "tile":
            pred_weights = np.tile(weight_tile, (pred_values.shape[0], 1, 1))[:, :, :, np.newaxis]
            pred_weights_reshaped = beo.patches_to_array(pred_weights, og_shape, tile_size)
            pred_weights_reshaped = pred_weights_reshaped
        elif merge_weights == "conf":
            pred_weights_reshaped = conf_reshaped
        elif merge_weights == "both":
            pred_weights = np.tile(weight_tile, (pred_values.shape[0], 1, 1))[:, :, :, np.newaxis]
            pred_weights_reshaped = beo.patches_to_array(pred_weights, og_shape, tile_size) * conf_reshaped
        elif merge_weights == "none":
            pred_weights_reshaped = np.ones_like(conf_reshaped, dtype="float32")
        else:
            raise ValueError(f"Unknown merge_weights method. Valid are: 'tile', 'conf', 'both', 'none'. Recieved: {merge_weights}")

        sx, ex, sy, ey = offset
        arr[sx:ex, sy:ey, idx] = pred_reshaped
        weights[sx:ex, sy:ey, idx] = pred_weights_reshaped

    arr = arr[:, :, :, 0]
    weights = weights[:, :, :, 0]

    weights_norm = (weights / np.sum(weights, axis=2, keepdims=True))

    merged = None
    if merge_method == "mean":
        merged = np.average(arr, axis=2, weights=weights_norm)[:, :, np.newaxis]

        if output_confidence:
            weights = np.average(weights, axis=2, weights=weights_norm)[:, :, np.newaxis]

    elif merge_method == "max_conf":
        mask = np.argmax(weights, axis=-1)[:, :, np.newaxis] == np.tile(
            np.arange(0, weights.shape[2]), weights.shape[0] * weights.shape[1]
        ).reshape(weights.shape[0], weights.shape[1], weights.shape[2])

        merged = np.ma.masked_array(arr, mask=~mask).max(axis=2).filled(0)[:, :, np.newaxis]

        if output_confidence:
            weights = np.ma.masked_array(weights, mask=~mask).max(axis=2).filled(0)[:, :, np.newaxis]

    elif merge_method == "median":
        merged = beo.weighted_median(arr, weights_norm)

        if output_confidence:
            weights = beo.weighted_median(weights, weights_norm)

    elif merge_method == "mad":
        merged = beo.mad_merge(arr, weights_norm)

        if output_confidence:
            weights = beo.mad_merge(weights, weights_norm)

    if output_variance:
        if output_confidence:
            return merged, weights, np.var(arr, axis=2, keepdims=True)
        else:
            return merged, np.var(arr, axis=2, keepdims=True)

    if output_confidence:
        return merged, weights
    
    return merged


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

    bands["QI"]["CLDPRB_20m"] = glob(f"{safe_folder}/GRANULE/*/QI_DATA/MSK_CLDPRB_20m.jp2")[0]
    bands["QI"]["CLDPRB_60m"] = glob(f"{safe_folder}/GRANULE/*/QI_DATA/MSK_CLDPRB_60m.jp2")[0]

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
