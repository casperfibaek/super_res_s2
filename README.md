# Super-resolution for Sentinel 2 files.

This repository provides a function to Super-sample the 20m bands of the Sentinel 2 imagery to 10m.

It works by using an inception res-net style Deep Learning model trained on 200 sites randomly selected around the globe. With at least 50% of locations containing urban areas. For each location, twelve training images were collected spread out across different seasons.

The model itself is trained by using the RGB bands to sharpen the NIR band. *First* the resampled NIR band is transposed to the mean values of the RGB bands, *secondly* the network supersamples the NIR band, and *thirdly* the network mean-matches the low resolution image to the generated high-resolution image. To super-sample the other bands, they are substituted with the NIR band. The model has been purposely made small to ensure easy deployment.


```python
# Setup
!pip install -i https://test.pypi.org/simple/ superres-casperfibaek==0.0.4
from superres_casperfibaek.super_res_s2 import super_sample

# Constants
YEAR = 2021
MONTHS = 1
AOI = [55.67576, 12.56902] # Copenhagen

# Example get Sentinel 2 data function.
data = get_data_from_latlng(AOI, year=YEAR, months=MONTHS)[0] 

# Fast is about 2.5 times faster and almost as good.
super_sampled = super_sample(data, method="fast", fit_data=False)
```

![Super-sampled bands: B05, B06, B07, B8A, B11, B12](./high_quality.png)

# super_sample
Super-sample a Sentinel 2 image. The source can either be a NumPy array of the bands, or a .safe file.

## Args:
`data` (_str_/_np.ndarray_): The image to supersample. Either .safe file or NumPy array. </br>

## Kwargs:
`indices` (_dict_): If the input is not a .safe file, a dictionary with the band names and the indices in the NumPy array must be proved. It comes in the form of { "B02": 0, "B03": 1, ... } (Default: **10m first, then 20m**) </br>
`method` (_str_): Either fast or accurate. If fast, uses less overlaps and weighted average merging. If accurate, uses more overlaps and the mad_merge algorithm (Default: **"fast"**) </br>
`fit_data` (_bool_): Should the deep learning model be fitted with the data? Improves accuracy, but takes around 1m to fit on colab. (Default: **True**) </br>
`verbose` (_bool_): If True, print statements will update on the progress (Default: **True**) </br>
`normalise` (_bool_): If the input data should be normalised. Leave this True, unless it has already been done. The model expects sentinel 2 l2a data normalised by dividing by 10000.0 (Default: **True**) </br>

## Returns:
(_np.ndarray_): A NumPy array with the supersampled data.

# Cite
Fibaek, C.S, Super-sample Sentinel 2, (2022), GitHub repository, https://github.com/casperfibaek/super_res_s2

Developed at the European Space Agency's Φ-lab.

# Build
python -m build; python -m twine upload --repository testpypi dist/*
