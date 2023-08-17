variable "subscription_id" {
  type        = string
  description = "The Azure subscription id."
} 

variable "client_id" {
  type        = string
  description = "The Azure client id."
} 

variable "client_secret" {
  type        = string
  description = "The Azure client secret."
  sensitive   = true
} 

variable "tenant_id" {
  type        = string
  description = "The Azure tenant id."
}

# General variables

variable "env" {
  type        = string
  description = "The environment."
  default     = "dev"
} 

variable "location" {
  type        = string
  description = "The name of the location (eastus,westus,etc.)."
  default     = "westus2"
} 

variable "prefix" {
  type        = string
  description = "The prefix for the resources."
}