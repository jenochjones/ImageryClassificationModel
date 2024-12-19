#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 14:14:19 2024

@author: jonjones
"""
import rasterio
import numpy as np
import joblib
import time

# Start the timer
start_time = time.time()

# File paths
input_raster_path = "./TifFiles/SalemBoundary_Masked.tif"
model_path = "./ImageryModel.pkl"
classified_raster_output_path = "./TifFiles/Salem_Classified.tif"

# Load the trained model
clf = joblib.load(model_path)
print(f"Model loaded from {model_path}")

# Load the input raster
with rasterio.open(input_raster_path) as src:
    profile = src.profile
    transform = src.transform
    crs = src.crs

    # Update the profile for the classified raster
    profile.update(count=1, dtype='uint8')

    # Open the output file for writing
    with rasterio.open(classified_raster_output_path, "w", **profile) as dst:

        # Process the raster in chunks (windows)
        for window_idx, window in src.block_windows(1):

            # Read the current window
            raster_window = src.read(window=window)  # Shape: (bands, rows, cols)

            # Reshape the window for prediction (pixels, bands)
            bands, rows, cols = raster_window.shape
            pixels = raster_window.reshape(bands, -1).T  # Shape: (pixels, bands)

            # Predict using the trained model
            print(f"Processing window {window}...")
            classified_pixels = clf.predict(pixels)

            # Reshape the classified data back to the window dimensions
            classified_window = classified_pixels.reshape(rows, cols)

            # Write the classified window to the output raster
            dst.write(classified_window.astype("uint8"), 1, window=window)

print(f"Classified raster saved to {classified_raster_output_path}")

# Stop the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Script execution time: {elapsed_time:.2f} seconds")

