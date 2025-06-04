import unittest
import numpy as np
import cv2
from preprocessing import Binarization, SkewCorrection, NoiseRemoval
import os

class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test images"""
        # Create a simple test image
        cls.test_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.putText(cls.test_image, 'Test', (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        
        # Create a skewed test image
        cls.skewed_image = np.zeros((100, 100), dtype=np.uint8)
        M = cv2.getRotationMatrix2D((50, 50), 15, 1.0)  # 15 degree rotation
        cls.skewed_image = cv2.warpAffine(cls.test_image, M, (100, 100))
        
        # Create a noisy test image
        cls.noisy_image = cls.test_image.copy()
        noise = np.random.normal(0, 25, cls.test_image.shape).astype(np.uint8)
        cls.noisy_image = cv2.add(cls.noisy_image, noise)

    def test_binarization_basic(self):
        """Test basic binarization"""
        result = Binarization.basic(self.test_image)
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertTrue(np.all(np.unique(result) == [0, 255]))

    def test_binarization_otsu(self):
        """Test Otsu's binarization"""
        # Test without Gaussian blur
        result1 = Binarization.otsu(self.test_image, with_gaussian=False)
        self.assertEqual(result1.shape, self.test_image.shape)
        self.assertTrue(np.all(np.unique(result1) == [0, 255]))

        # Test with Gaussian blur
        result2 = Binarization.otsu(self.test_image, with_gaussian=True)
        self.assertEqual(result2.shape, self.test_image.shape)
        self.assertTrue(np.all(np.unique(result2) == [0, 255]))

    def test_binarization_adaptive(self):
        """Test adaptive binarization methods"""
        # Test adaptive mean
        result1 = Binarization.adaptive_mean(self.test_image)
        self.assertEqual(result1.shape, self.test_image.shape)
        self.assertTrue(np.all(np.unique(result1) == [0, 255]))

        # Test adaptive Gaussian
        result2 = Binarization.adaptive_gaussian(self.test_image)
        self.assertEqual(result2.shape, self.test_image.shape)
        self.assertTrue(np.all(np.unique(result2) == [0, 255]))

    def test_binarization_yannihorne(self):
        """Test Yannihorne binarization"""
        result = Binarization.yannihorne(self.test_image)
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertTrue(np.all(np.unique(result) == [0, 255]))

    def test_binarization_niblack(self):
        """Test Niblack binarization"""
        result = Binarization.niblack(self.test_image)
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertTrue(np.all(np.unique(result) == [0, 255]))

    def test_skew_correction_boxes(self):
        """Test boxes-based skew correction"""
        result = SkewCorrection.boxes(self.skewed_image)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.shape), 2)  # Should be 2D image

    def test_skew_correction_hough(self):
        """Test Hough transform skew correction"""
        result = SkewCorrection.hough_transform(self.skewed_image)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.shape), 2)

    def test_skew_correction_moments(self):
        """Test moments-based skew correction"""
        result = SkewCorrection.moments(self.skewed_image)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.shape), 2)

    def test_noise_removal_filters(self):
        """Test various noise removal filters"""
        # Test mean filter
        result1 = NoiseRemoval.mean_filter(self.noisy_image)
        self.assertEqual(result1.shape, self.noisy_image.shape)

        # Test Gaussian filter
        result2 = NoiseRemoval.gaussian_filter(self.noisy_image)
        self.assertEqual(result2.shape, self.noisy_image.shape)

        # Test median filter
        result3 = NoiseRemoval.median_filter(self.noisy_image)
        self.assertEqual(result3.shape, self.noisy_image.shape)

        # Test conservative filter
        result4 = NoiseRemoval.conservative_filter(self.noisy_image)
        self.assertEqual(result4.shape, self.noisy_image.shape)

    def test_noise_removal_advanced(self):
        """Test advanced noise removal methods"""
        # Test Laplacian filter
        result1 = NoiseRemoval.laplacian_filter(self.noisy_image)
        self.assertEqual(result1.shape, self.noisy_image.shape)

        # Test frequency filter
        result2 = NoiseRemoval.frequency_filter(self.noisy_image)
        self.assertEqual(result2.shape, self.noisy_image.shape)

        # Test unsharp filter
        result3 = NoiseRemoval.unsharp_filter(self.noisy_image)
        self.assertEqual(result3.shape, self.noisy_image.shape)

if __name__ == '__main__':
    unittest.main() 