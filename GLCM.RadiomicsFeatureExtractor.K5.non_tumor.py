import radiomics
import numpy as np
import SimpleITK as sitk
import radiomics.featureextractor
import os
import six
from os import path
import pandas as pd

def featureExtractor(fileId):
  imagePath = './dataset/BraTS2021_Training_Data/%s/%s_flair.nii.gz' % (fileId, fileId)
  image = sitk.ReadImage(imagePath)
  maskPath ='./dataset/BraTS2021_Training_Data/%s/%s_kernel5_non_tumor.nii.gz' % (fileId, fileId)
  mask = sitk.ReadImage(maskPath)

  kernel = 5
  settings = {}
  settings['kernelRadius'] = kernel
  settings['maskedKernel'] = False
  settings['voxelBatch'] = 10
  extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(**settings)
  extractor.disableAllFeatures()
  extractor.enableFeatureClassByName('glcm')

  featureMap = extractor.execute(image, mask, voxelBased=True)

  for featureName, featureValue in six.iteritems(featureMap):
    if isinstance(featureValue, sitk.Image):
      fileFolder = './dataset/glcm/kernel5-radius5/non_tumor/%s' % (fileId)
      if path.exists(fileFolder) == False:
        os.mkdir(fileFolder)
      sitk.WriteImage(featureValue, '%s/%s.nrrd' % (fileFolder, featureName))
      print('Computed %s, stored as "%s/%s.nrrd"' % (featureName, fileFolder, featureName))
    # else:
    #   print('%s: %s' % (featureName, featureValue))

monitorFilePath = './dataset/glcm/kernel5-radius5/non_tumor.monitor.csv'
monitor = pd.read_csv(monitorFilePath, index_col='no')
for i, row in monitor.iterrows():
  if row['done'] != 0:
    continue
  fileId = row['file']
  print('Starting %s' % (fileId))
  featureExtractor(fileId)
  monitor.at[i, 'done'] = 1
  monitor.to_csv(monitorFilePath)