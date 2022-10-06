# Super-resolution for Sentinel 2 files.

### *This library presents a special approach to sharpening imagery, where areas are only sharpened if the model is confident in doing so.*

It works by using an inception res-net style Deep Learning model trained on 1000 sites randomly selected around the globe.
The sites represent at least 25 samples within each Köppen-Geiger climate zone and at least one image for each city in the world with at least 1 million inhabitants. For each location, three training mosaics were collected spread out across different seasons resulting in a total of 3000 mosaics.

The model itself is trained by using the RGB bands to sharpen the downsampled NIR band. *First* the resampled NIR band is transposed to the mean values of the RGB bands, *secondly* the network super-samples the transposed NIR band, and *thirdly* the network mean-matches the low-resolution image to the generated high-resolution image. To super-sample the other bands, they are substituted with the NIR band. The model has been purposely made small to ensure easy deployment, and the methodology is quite conservative in its estimates to ensure that no wild predictions are made.

The package aims to be a drop-in replacement for arrays sharpened with the bilinear method and _should_ provide a minor improvement in downstream model accuracy. Please look at the example colab or investigate the results in their full resolution.

**Dependencies** </br>
`buteo`(https://casperfibaek.github.io/buteo/) </br>
`tensorflow` (https://www.tensorflow.org/) </br>

**Installation** </br>
`pip install s2super` </br>

**Quickstart**
```python
# Setup
from s2super import super_sample

# Constants
YEAR = 2021
MONTHS = 1
AOI = [0.039611, -51.169216] # Macapá

# Example get Sentinel 2 data function.
data = get_data_from_latlng(AOI, year=YEAR, months=MONTHS)[0] 

# Fast is about 2.5 times faster and almost as good.
super_sampled = super_sample(data, method="fast", fit_data=False)
```

Example colab notebook [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/casperfibaek/super_res_s2/blob/master/example.ipynb)

|![Super-sampled band 12. TOPLEFT: RGB, TOPRIGHT: CONFIDENCE, BOTLEFT: UNSHARP, BOTRIGHT:SHARP](https://github.com/casperfibaek/super_res_s2/raw/main/b12_130dpi_zoomout.png)|
|:--:|
| *Sentinel 2 Band 12 (SWIR2) Sharpnened. </br> Clockwise from topleft: RGB, Confidence, B12-unsharp, B12-sharpened (Beirut - Lebanon, 2022)* |
|![Super-sampled band 12 Zoomed 1. TOPLEFT: RGB, TOPRIGHT: CONFIDENCE, BOTLEFT: UNSHARP, BOTRIGHT:SHARP](https://github.com/casperfibaek/super_res_s2/raw/main/b8a_130dpi_zoomin_1.png)|
| *Sentinel 2 Band 8A (Narrow-NIR) Sharpnened. Zoomed in. </br> Clockwise from topleft: RGB, Confidence, B8A-unsharp, B8A-sharpened (Aley - Lebanon, 2022)* |
![Super-sampled band 12 Zoomed 2. TOPLEFT: RGB, TOPRIGHT: CONFIDENCE, BOTLEFT: UNSHARP, BOTRIGHT:SHARP](https://github.com/casperfibaek/super_res_s2/raw/main/b05_130dpi_zoomin_2.png)
| *Sentinel 2 Band 5 (Red Edge 1) Sharpnened. Zoomed in. </br> Clockwise from topleft: RGB, Confidence, B5-unsharp, B5-sharpened (Rafic Hariri Airport - Lebanon, 2022)* |

# super_sample
Super-sample a Sentinel 2 image. The source can either be a NumPy array of the bands, or a .safe file.

## Args:
`data` (_str_/_np.ndarray_): The image to supersample. Either .safe file or NumPy array. </br>

## Kwargs:
`indices` (_dict_): If the input is not a .safe file, a dictionary with the band names and the indices in the NumPy array must be proved. It comes in the form of { "B02": 0, "B03": 1, ... } (Default: **10m first, then 20m**) </br>
`method` (_str_): Either fast or accurate. If fast, uses less overlaps and weighted average merging. If accurate, uses more overlaps and the mad_merge algorithm (Default: **"fast"**) </br>
`fit_data` (_bool_): Should the deep learning model be fitted with the data? Improves accuracy, but takes around 1m to fit on colab. (Default: **True**) </br>
`fit_epochs` (_int_): If the model is refitted, for how many epochs should it run? (Default: **5**) </br>
`verbose` (_bool_): If True, print statements will update on the progress (Default: **True**) </br>
`normalise` (_bool_): If the input data should be normalised. Leave this True, unless it has already been done. The model expects sentinel 2 l2a data normalised by dividing by 10000.0 (Default: **True**) </br>
`preloaded_model` (_None/tf.model_): Allows preloading the model, useful if applying the super_sampling function within a loop. (Default: **None**) </br>

## Returns:
(_np.ndarray_): A NumPy array with the supersampled data.

# Cite
Fibaek, C.S, Super-sample Sentinel 2, (2022), GitHub repository, https://github.com/casperfibaek/super_res_s2

Developed at the European Space Agency's Φ-lab.

# Build
python -m build; python -m twine upload dist/*

# Cuda-setup
conda install -c nvidia cuda-python
conda install -c conda-forge cudnn
