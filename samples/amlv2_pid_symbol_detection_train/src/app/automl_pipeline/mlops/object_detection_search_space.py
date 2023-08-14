"""
This file defines the setting for automl image object detection settings
"""

object_detection_training_parameters = [
    "learning_rate",
    "learning_rate_scheduler",
    "model_name",
    "number_of_epochs",
    "optimizer",
    "step_lr_gamma",
    "step_lr_step_size",
    "training_batch_size",
    "validation_batch_size",
    "warmup_cosine_lr_cycles",
    "warmup_cosine_lr_warmup_epochs",
    "weight_decay",
    "box_detections_per_image",
    "box_score_threshold",
    "image_size",
    "max_size",
    "min_size",
    "model_size",
    "multi_scale",
    "nms_iou_threshold",
    "tile_grid_size",
    "tile_overlap_ratio",
    "tile_predictions_nms_threshold",
    "validation_iou_threshold",
]
