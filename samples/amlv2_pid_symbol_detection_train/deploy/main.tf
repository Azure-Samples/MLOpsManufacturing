locals {
  resource_name_prefix_dashes    = "${var.prefix}-${var.env}-${var.location}"
  resource_name_prefix_no_dashes = "${var.prefix}${var.env}${var.location}"
}

resource "azurerm_resource_group" "rg" {
  name     = "${local.resource_name_prefix_dashes}-rg"
  location = var.location
}

# Dependent resources for Azure Machine Learning
resource "azurerm_application_insights" "app_insights" {
  name                = "${local.resource_name_prefix_dashes}-appi"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  application_type    = "web"
}

resource "azurerm_key_vault" "kv" {
  name                     = "${local.resource_name_prefix_no_dashes}kv"
  location                 = azurerm_resource_group.rg.location
  resource_group_name      = azurerm_resource_group.rg.name
  tenant_id                = var.tenant_id
  sku_name                 = "premium"
  purge_protection_enabled = true
}

resource "azurerm_storage_account" "sa" {
  name                            = "${local.resource_name_prefix_no_dashes}st"
  location                        = azurerm_resource_group.rg.location
  resource_group_name             = azurerm_resource_group.rg.name
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  allow_nested_items_to_be_public = false 
}

resource "azurerm_container_registry" "cr" {
  name                = "${local.resource_name_prefix_no_dashes}cr"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "Premium"
  admin_enabled       = true
}

# Machine Learning workspace
resource "azurerm_machine_learning_workspace" "aml_workspace" {
  name                          = "${local.resource_name_prefix_dashes}-mlw"
  location                      = azurerm_resource_group.rg.location
  resource_group_name           = azurerm_resource_group.rg.name
  application_insights_id       = azurerm_application_insights.app_insights.id
  key_vault_id                  = azurerm_key_vault.kv.id
  storage_account_id            = azurerm_storage_account.sa.id
  container_registry_id         = azurerm_container_registry.cr.id
  public_network_access_enabled = true

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_machine_learning_compute_cluster" "cpu_cluster" {
  name                          = "cpu-cluster"
  location                      = azurerm_resource_group.rg.location
  vm_priority                   = "Dedicated"
  vm_size                       = "Standard_E4as_v4"
  machine_learning_workspace_id = azurerm_machine_learning_workspace.aml_workspace.id

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = 2
    scale_down_nodes_after_idle_duration = "PT30S" # 30 seconds
  }

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_machine_learning_compute_cluster" "gpu_cluster" {
  name                          = "gpu-cluster"
  location                      = azurerm_resource_group.rg.location
  vm_priority                   = "Dedicated"
  vm_size                       = "Standard_NC64as_T4_v3"
  machine_learning_workspace_id = azurerm_machine_learning_workspace.aml_workspace.id

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = 25
    scale_down_nodes_after_idle_duration = "PT30S" # 30 seconds
  }

  identity {
    type = "SystemAssigned"
  }
}