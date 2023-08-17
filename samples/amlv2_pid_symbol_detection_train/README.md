# Overview

This example demonstrates how to use Azure ML Python SDK v2 to train a symbol detection (yoloV5 model) using P&ID synthetic dataset.

[Sample project documentation](./docs/README.md).

__What does this sample demonstrate__:

This sample allows the creation of an automated AML workflow that orchestrate the next six main steps:

1. **Data Aggregation:** This step involves aggregating the data required for training the machine learning model.

1. **Data Split:** This step splits data to training and validation sets prior to training.

1. **Register Data Asset:** This step registers the aggregated data as a new version of the training data asset.

1. **Training the Machine Learning Model:** In this step, the machine learning model is trained using the aggregated data.

1. **Registering the Model:** After a model has been trained, it is necessary to register it by creating an artifact that can be used for deployment.

1. **Tagging the Model:** Finally, the best performing model is tagged so that it can be easily identified and used for future tasks.

By following these key steps, the symbol detection pipeline can effectively automate the machine learning workflow, leading to efficient and effective results.

__What doesn't this sample demonstrate__:

* Model inferencing
