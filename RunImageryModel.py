#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 14:14:19 2024

@author: jonjones
"""

import numpy as np
import rasterio
import joblib

# File paths
input_tif_file = "./Imagery/SalemBoundary_Masked.tif"
model_file = "./ImageryModel.joblib"
output_tif_file = "./Imagery/SalemBoundary_Classified.tif"

# Load the trained model
clf = joblib.load(model_file)

# Load the raster data
with rasterio.open(input_tif_file) as src:
    raster_data = src.read(1)
    transform = src.transform
    nodata_value = src.nodata
    meta = src.meta

# Ensure nodata_value is a valid integer
if nodata_value is None:
    nodata_value = -9999  # Assign a default nodata value

# Prepare the raster for classification
buffer = 1  # Neighborhood buffer size
classified_data = np.full(raster_data.shape, nodata_value, dtype=np.int32)

# Classify each pixel
for i in range(raster_data.shape[0]):
    for j in range(raster_data.shape[1]):
        if raster_data[i, j] == nodata_value:
            continue

        # Extract the neighborhood around the pixel
        neighbors = raster_data[max(0, i - buffer): i + buffer + 1, max(0, j - buffer): j + buffer + 1]

        # Ensure neighbors array matches the expected input size
        if neighbors.shape[0] < 2 * buffer + 1 or neighbors.shape[1] < 2 * buffer + 1:
            continue

        neighbors = neighbors.flatten()

        # If the neighborhood contains nodata values, skip
        if nodata_value is not None and np.any(neighbors == nodata_value):
            continue

        # Pad the neighborhood to match the expected feature size
        expected_features = clf.n_features_in_
        if len(neighbors) < expected_features:
            neighbors = np.pad(neighbors, (0, expected_features - len(neighbors)), constant_values=nodata_value)

        # Predict the class using the trained model
        predicted_class = clf.predict([neighbors])[0]
        classified_data[i, j] = predicted_class

# Save the classified raster
meta.update(dtype=rasterio.int32, nodata=nodata_value)

with rasterio.open(output_tif_file, "w", **meta) as dst:
    dst.write(classified_data, 1)

print(f"Classified raster saved to {output_tif_file}")