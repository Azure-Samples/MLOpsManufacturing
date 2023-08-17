import os
import sys
import unittest
import tempfile
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "app",
        "pipeline_steps",
        "data_splitter_src",
    )
)
from app.pipeline_steps.data_splitter_src.main import main


class TestMain(unittest.TestCase):
    def setUp(self):
        self._input_data_path = os.path.join(os.path.dirname(__file__), "data")

    def test_main(self):
        temp_out_dir = tempfile.TemporaryDirectory()
        train_output_path = str(Path(temp_out_dir.name) / "train")
        val_output_path = str(Path(temp_out_dir.name) / "val")

        main(
            self._input_data_path,
            train_output_path,
            val_output_path,
            "",
            "",
        )

        self.assertTrue(Path(Path(train_output_path) / "annotations.jsonl").is_file())
        self.assertTrue(Path(Path(train_output_path) / "MLTable").is_file())
        self.assertTrue(Path(Path(val_output_path) / "annotations.jsonl").is_file())
        self.assertTrue(Path(Path(val_output_path) / "MLTable").is_file())

        temp_out_dir.cleanup()
