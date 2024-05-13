import rasterio
import numpy as np

# open the GeoTIFF file using rasterio (you can update location of tif file below)
with rasterio.open('sample_rgb.tif') as src:
    red = src.read(1)
    green = src.read(2)
    blue = src.read(3)
    
    # new band (B1 - B2) / (B1 + B2 + B3) calculations
    new_band = (red - green) / (red + green + blue)
    
    # standard deviation of the values calculation using numPy
    std_dev = np.std([red, green, blue], axis=0)

    # creating a profile for the new GeoTIFF files & write new band
    profile = src.profile
    profile.update(count=1)
    with rasterio.open('new_band.tif', 'w', **profile) as dst:
        dst.write(new_band, 1)
        
    # updating the profile for the std deviation GeoTIFF then writing std deviation to the file
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open('std_dev.tif', 'w', **profile) as dst:
        dst.write(std_dev, 1)