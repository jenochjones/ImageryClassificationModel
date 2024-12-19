#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 14:00:54 2024

@author: jonjones
"""
import rasterio
from rasterio.features import geometry_mask  # Correctly import geometry_mask
import geopandas as gpd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# File paths
raster_path = "./TifFiles/SalemBoundary_Masked.tif"
shapefile_path = "./Shapefiles/Imagery_Classification_Training_Data.shp"
output_model_path = "./ImageryModel.pkl"

# Load raster data
with rasterio.open(raster_path) as src:
    raster = src.read()  # Shape: (bands, rows, cols)
    transform = src.transform
    crs = src.crs
    profile = src.profile
    profile.update(count=1, dtype='uint8')  # Update for single-band classified output

# Flatten raster for easier processing
rows, cols = raster.shape[1], raster.shape[2]
raster_data = raster.reshape(raster.shape[0], -1).T  # Shape: (pixels, bands)

# Load shapefile
shapefile = gpd.read_file(shapefile_path)
shapefile = shapefile.to_crs(crs.to_string())  # Ensure the CRS matches the raster

# Create a mask of labeled pixels
labels = np.zeros((rows, cols), dtype=np.int32)  # Initialize label array

# Burn labels into the raster space
for _, row in shapefile.iterrows():
    geom = row.geometry
    label_id = row["ID"]  # Assumes 'ID' is the field for classification labels
    mask = geometry_mask([geom], out_shape=(rows, cols), transform=transform, invert=True)
    labels[mask] = label_id

# Flatten the label array
label_data = labels.flatten()

# Extract labeled pixel data (non-zero labels)
mask = label_data > 0
X = raster_data[mask]  # Features (pixels with non-zero labels)
y = label_data[mask]  # Labels

# Split into train and test sets (90/10 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(clf, output_model_path)
print(f"Model saved to {output_model_path}")
