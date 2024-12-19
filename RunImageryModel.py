#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 14:14:19 2024

@author: jonjones
"""



# File paths
input_tif_file = "./TifFiles/SalemBoundary_Masked.tif"
model_file = "./ImageryModel.joblib"
output_tif_file = "./TifFiles/SalemBoundary_Classified.tif"

import rasterio
import numpy as np
import joblib

# File paths
input_raster_path = "./TifFiles/SalemBoundary_Masked.tif"
model_path = "./ImageryModel.pkl"
classified_raster_output_path = "./TifFiles/Salem_Classified.tif"

# Load the trained model
clf = joblib.load(model_path)
print(f"Model loaded from {model_path}")

# Load the input raster
with rasterio.open(input_raster_path) as src:
    raster = src.read()  # Shape: (bands, rows, cols)
    transform = src.transform
    crs = src.crs
    profile = src.profile

# Get raster dimensions
bands, rows, cols = raster.shape

# Ensure the raster is large enough for a 500x500 crop
if rows < 500 or cols < 500:
    raise ValueError("The input raster is smaller than 500x500 cells.")

# Compute the center of the raster
center_row = rows // 2
center_col = cols // 2

# Determine the crop window for a 500x500 square
start_row = max(center_row - 250, 0)
start_col = max(center_col - 250, 0)
end_row = start_row + 500
end_col = start_col + 500

# Ensure the window is within bounds
if end_row > rows or end_col > cols:
    raise ValueError("The 500x500 crop exceeds raster bounds.")

# Extract the 500x500 square
raster_crop = raster[:, start_row:end_row, start_col:end_col]  # Shape: (bands, 500, 500)

# Reshape the crop for prediction (pixels, bands)
pixels = raster_crop.reshape(bands, -1).T  # Shape: (pixels, bands)

# Predict using the trained model
classified_pixels = clf.predict(pixels)

# Reshape the classified data back to 500x500
classified_crop = classified_pixels.reshape(500, 500)

# Save the classified crop to a new GeoTIFF
profile.update(height=500, width=500, count=1, dtype='uint8', transform=rasterio.windows.transform(
    rasterio.windows.Window(start_col, start_row, 500, 500), transform
))

with rasterio.open(classified_raster_output_path, "w", **profile) as dst:
    dst.write(classified_crop.astype("uint8"), 1)

print(f"Classified raster saved to {classified_raster_output_path}")
