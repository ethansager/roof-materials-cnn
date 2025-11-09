# Test Dataset Module for roof-materials-cnn

import unittest
from src.data.dataset import YourDatasetClass  # Replace with your actual dataset class
from torchvision import transforms

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.transform = transforms.Compose([
            # Add your transformations here
        ])
        self.dataset = YourDatasetClass(root='path/to/data', transform=self.transform)

    def test_length(self):
        self.assertEqual(len(self.dataset), expected_length)  # Replace expected_length with the actual value

    def test_get_item(self):
        image, target = self.dataset[0]
        self.assertIsNotNone(image)
        self.assertIsNotNone(target)

    def test_transform(self):
        image, target = self.dataset[0]
        transformed_image = self.transform(image)
        self.assertEqual(transformed_image.size(), expected_size)  # Replace expected_size with the actual value

if __name__ == '__main__':
    unittest.main()