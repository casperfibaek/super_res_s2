import os
import numpy as np
import tensorflow as tf
from numba import jit, prange
from glob import glob


@jit(nopython=True, parallel=True, nogil=True, inline="always")
def weighted_median(arr, weight_arr):

    ret_arr = np.empty((arr.shape[0], arr.shape[1], 1), dtype="float32")

    for x in prange(arr.shape[0]):
        for y in range(arr.shape[1]):
            values = arr[x, y].flatten()
            weights = weight_arr[x, y].flatten()

            sort_mask = np.argsort(values)
            sorted_data = values[sort_mask]
            sorted_weights = weights[sort_mask]
            cumsum = np.cumsum(sorted_weights)
            intersect = (cumsum - 0.5 * sorted_weights) / cumsum[-1]
            ret_arr[x, y, 0] = np.interp(0.5, intersect, sorted_data)

    return ret_arr


@jit(nopython=True, parallel=True, nogil=True, inline="always")
def mad_merge(arr, weight_arr, mad_dist=1.0):

    ret_arr = np.empty((arr.shape[0], arr.shape[1], 1), dtype="float32")

    for x in prange(arr.shape[0]):
        for y in prange(arr.shape[1]):
            values = arr[x, y].flatten()
            weights = weight_arr[x, y].flatten()

            sort_mask = np.argsort(values)
            sorted_data = values[sort_mask]
            sorted_weights = weights[sort_mask]
            cumsum = np.cumsum(sorted_weights)
            intersect = (cumsum - 0.5 * sorted_weights) / cumsum[-1]
            
            median = np.interp(0.5, intersect, sorted_data)
            mad = np.median(np.abs(median - values))

            if mad == 0.0:
                ret_arr[x, y, 0] = median
                continue
            
            new_weights = np.zeros_like(sorted_weights)
            for z in range(sorted_data.shape[0]):
                new_weights[z] = 1.0 - (np.minimum(np.abs(sorted_data[z] - median) / (mad * mad_dist), 1))

            cumsum = np.cumsum(new_weights)
            intersect = (cumsum - 0.5 * new_weights) / cumsum[-1]
            
            ret_arr[x, y, 0] = np.interp(0.5, intersect, sorted_data)

    return ret_arr


def get_offsets(size, number_of_offsets=3):
    assert number_of_offsets <= 9, "Number of offsets must be nine or less"
    offsets = [[0, 0]]

    if number_of_offsets == 0:
        return offsets

    mid = size // 2
    low = mid // 2
    high = mid + low

    additional_offsets = [
        [mid, mid],
        [0, mid],
        [mid, 0],
        [0, low],
        [low, 0],
        [high, 0],
        [0, high],
        [low, low],
        [high, high],
    ]

    offsets += additional_offsets[:number_of_offsets]

    return offsets


def array_to_blocks(arr, tile_size, offset=[0, 0]):
    blocks_y = (arr.shape[0] - offset[1]) // tile_size
    blocks_x = (arr.shape[1] - offset[0]) // tile_size

    cut_y = -((arr.shape[0] - offset[1]) % tile_size)
    cut_x = -((arr.shape[1] - offset[0]) % tile_size)

    cut_y = None if cut_y == 0 else cut_y
    cut_x = None if cut_x == 0 else cut_x

    og_coords = [offset[1], cut_y, offset[0], cut_x]
    og_shape = list(arr[offset[1] : cut_y, offset[0] : cut_x].shape)

    reshaped = arr[offset[1] : cut_y, offset[0] : cut_x].reshape(
        blocks_y,
        tile_size,
        blocks_x,
        tile_size,
        arr.shape[2],
    )

    swaped = reshaped.swapaxes(1, 2)
    blocks = swaped.reshape(-1, tile_size, tile_size, arr.shape[2])

    return blocks, og_coords, og_shape


def blocks_to_array(blocks, og_shape, tile_size, offset=[0, 0]):
    with np.errstate(invalid="ignore"):
        target = np.empty(og_shape, dtype="float32") * np.nan

    target_y = ((og_shape[0] - offset[1]) // tile_size) * tile_size
    target_x = ((og_shape[1] - offset[0]) // tile_size) * tile_size

    cut_y = -((og_shape[0] - offset[1]) % tile_size)
    cut_x = -((og_shape[1] - offset[0]) % tile_size)

    cut_x = None if cut_x == 0 else cut_x
    cut_y = None if cut_y == 0 else cut_y

    reshape = blocks.reshape(
        target_y // tile_size,
        target_x // tile_size,
        tile_size,
        tile_size,
        blocks.shape[3],
        1,
    )

    swap = reshape.swapaxes(1, 2)

    destination = swap.reshape(
        (target_y // tile_size) * tile_size,
        (target_x // tile_size) * tile_size,
        blocks.shape[3],
    )

    target[offset[1] : cut_y, offset[0] : cut_x] = destination

    return target


def get_kernel_weights(tile_size=64, edge_distance=5, epsilon=1e-7):
    arr = np.empty((tile_size, tile_size), dtype="float32")
    max_dist = edge_distance * 2
    for y in range(0, arr.shape[0]):
        for x in range(0, arr.shape[1]):
            val_y_top = max(edge_distance - y, 0.0)
            val_y_bot = max((1 + edge_distance) - (tile_size - y), 0.0)
            val_y = val_y_top + val_y_bot

            val_x_lef = max(edge_distance - x, 0.0)
            val_x_rig = max((1 + edge_distance) - (tile_size - x), 0.0)
            val_x = val_x_lef + val_x_rig

            val = (max_dist - abs(val_y + val_x)) / max_dist

            if val <= 0.0:
                val = epsilon
            
            arr[y, x] = val

    return arr


def get_overlaps(arr, tile_size, number_of_offsets=3, border_check=True):
    overlaps = []
    offsets = []
    shapes = []

    calc_borders = get_offsets(tile_size, number_of_offsets=number_of_offsets)
    if border_check:
        calc_borders.append([arr.shape[1] - tile_size, 0])
        calc_borders.append([0, arr.shape[0] - tile_size])
        calc_borders.append([arr.shape[1] - tile_size, arr.shape[0] - tile_size])

    for offset in calc_borders:
        blocks, og_coords, og_shape = array_to_blocks(arr, tile_size, offset)

        shapes.append(og_shape)
        offsets.append(og_coords)
        overlaps.append(blocks)

    return overlaps, offsets, shapes

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
        overlap, _, _ = get_overlaps(data, tile_size, number_of_offsets=number_of_offsets, border_check=borders)
        overlaps.append(overlap)

    _, offsets, shapes = get_overlaps(data_output_proxy, tile_size, number_of_offsets=number_of_offsets, border_check=borders)

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
        pred_reshaped = blocks_to_array(predicted, og_shape, tile_size)
       
        pred_weights = np.tile(weight_tile, (predicted.shape[0], 1, 1))[:, :, :, np.newaxis]
        pred_weights_reshaped = blocks_to_array(pred_weights, og_shape, tile_size)

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