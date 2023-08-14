import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "app"))
from automl_pipeline.mlops.component_repository import ComponentsRepository


class TestComponentRepository(unittest.TestCase):
    def test_defined_components(self):
        self.assertTrue(ComponentsRepository.data_aggregation)
        self.assertTrue(ComponentsRepository.data_split)
        self.assertTrue(ComponentsRepository.register_data_asset)
        self.assertTrue(ComponentsRepository.register)
        self.assertTrue(ComponentsRepository.tag)
