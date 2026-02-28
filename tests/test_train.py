import numpy as np 
import pytest 


# Mini version of parameters from train.py
DATASET_MEAN = 0.5147
DATASET_STD = 0.2536

# Mini version of custom_preprocessing function
def custom_preprocessing(img):
    img = img / 255.0
    img = (img - DATASET_MEAN) / DATASET_STD
    return img

# Test 1: Make sure custom_preprocessing does not change image shape
def test_custom_preprocessing_shape():
    dummy_img = np.ones((48, 48, 1)) * 128
    result = custom_preprocessing(dummy_img)
    
    assert result.shape == (48, 48, 1)


# Test 2: Check if normalization actually scales values down
def test_normalization_scale():
    dummy_img = np.ones((48, 48, 1)) * 128
    result = custom_preprocessing(dummy_img)
    
    assert result.mean() < 1

    
def test_standardization_values():
    dummy_img = np.ones((48, 48, 1)) * 255.0
    result = custom_preprocessing(dummy_img)
    expected = 1.91364353
    assert abs(result.mean() - expected) < 1e-5 
    