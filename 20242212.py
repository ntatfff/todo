from __future__ import print_function

import logging
import os

import SimpleITK as sitk
import six
import numpy as np

import radiomics
from radiomics import featureextractor, getFeatureClasses
from radiomics import glcm
from tqdm import tqdm

radiomics.setVerbosity(logging.INFO)
radiomics.progressReporter = tqdm

imageSize = (3, 3, 3)

mask_data = np.zeros(imageSize, dtype=int)
mask_data[1,1,1] = 1
mask_data[2,2,2] = 1
mask = sitk.GetImageFromArray(mask_data)

image_data = [[[10, 20, 30], 
              [40, 50, 60], 
              [70, 80, 90]],
              [[15, 25, 35], 
              [45, 55, 65], 
              [75, 85, 95]],
              [[19, 29, 39], 
              [49, 59, 69], 
              [79, 89, 99]]]
image = sitk.GetImageFromArray(image_data)

extractor = featureextractor.RadiomicsFeatureExtractor()

features = extractor.execute(image, mask, voxelBased=True)
sitk.GetArrayFromImage(features['original_firstorder_Energy'])