"""
Pipeline steps to be executed in an Azure ML Workspace
"""
from azure.ai.ml import load_component
import os


_base_component_path = os.path.join(os.path.dirname(__file__), "..", "components")


class ComponentsRepository:
    """
    Component that performs pre-training data aggregation on label data
    """

    data_aggregation = load_component(
        source=os.path.join(_base_component_path, "data_aggregation.yml")
    )
    """
    Component that performs pre-training data aggregation on label data
    """
    data_split = load_component(
        source=os.path.join(_base_component_path, "data_split.yml")
    )
    """
    Component that registering the aggregated training data asset
    """
    register_data_asset = load_component(
        source=os.path.join(_base_component_path, "register_data_asset.yml")
    )
    """
    Component that performs registration of the model
    """
    register = load_component(source=os.path.join(_base_component_path, "register.yml"))
    """
    Component that performs tagging to the "best" model
    """
    tag = load_component(source=os.path.join(_base_component_path, "tag.yml"))
