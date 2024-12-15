#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 14:00:54 2024

@author: jonjones
"""

import numpy as np
import rasterio
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from rasterio.features import geometry_mask
from shapely.geometry import box

# File paths
tif_file = "./Imagery/SalemBoundary_Masked.tif"
shapefile = "./Shapefiles/Imagery_Classification_Training_Data.shp"
model_output_file = "./ImageryModel.joblib"

# Load the raster data
with rasterio.open(tif_file) as src:
    raster_data = src.read(1)
    transform = src.transform
    nodata_value = src.nodata

# Load the shapefile
gdf = gpd.read_file(shapefile)

# Ensure the shapefile is in the same CRS as the raster
gdf = gdf.to_crs(crs=src.crs)

# Extract features and labels
def extract_features_and_labels(raster, gdf, transform, buffer=1):
    features = []
    labels = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        class_id = row.ID  # Ensure the ID field corresponds to the classification

        # Create a mask of the polygon
        mask = geometry_mask([geom], transform=transform, invert=True, out_shape=raster.shape)
        indices = np.argwhere(mask)

        for index in indices:
            i, j = index
            # Extract the neighborhood around the pixel
            neighbors = raster[max(0, i - buffer): i + buffer + 1, max(0, j - buffer): j + buffer + 1].flatten()

            # Exclude nodata values
            if nodata_value is not None and np.any(neighbors == nodata_value):
                continue

            features.append(neighbors)
            labels.append(class_id)

    return np.array(features), np.array(labels)

features, labels = extract_features_and_labels(raster_data, gdf, transform)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))

# Save the model
joblib.dump(clf, model_output_file)

print(f"Trained model saved to {model_output_file}")

