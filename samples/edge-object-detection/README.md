# Edge Object Detection <!-- omit in toc -->

This repository shows you how to do object detection on edge devices and route the detection results to the cloud!
The repository includes pipelines that fully automate the azure resource deployments
as well as the deployment of edge modules to do the object detection.

This repository uses an example use case of detecting a truck driving on a highway,
but by providing your own video (or live feed) you can detect whatever you please!

The [Getting Started](#getting-started) section of this document will walk you through how to set this up in your environment

## Sections <!-- omit in toc -->

- [Problem Summary](#problem-summary)
- [Solution Summary](#solution-summary)
- [Products/Technologies/Languages Used](#productstechnologieslanguages-used)
  - [Products and Technologies](#products-and-technologies)
  - [Languages](#languages)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Contributions](#contributions)
  - [Documentation](#documentation)
  - [Projects](#projects)
  - [Prerequisites](#prerequisites)
  - [Environment Setup](#environment-setup)
  - [Linting](#linting)

## Problem Summary

TODO: Finish this

## Solution Summary

TODO: finish this

## Products/Technologies/Languages Used

### Products and Technologies

- Azure IoT Hub
- Azure Log Analytics and App Insights
- Azure KeyVault
- Azure Media Services
- Azure Video Analyzer (previously LVA)
- Bicep

### Languages

- Python
- Bash (for some deployments)

## Architecture

The [Architecture](./docs/architecture.md) file shows a high level architecture
of the solution and the components used.

## Getting Started

The [Getting Started](./GettingStarted.md) file explains how to
get started and deploy the solution to your own environment!

## Contributions

If you want to develop your own business logic for edge object detection,
these sections will provide more information on how that can be accomplished!

### Documentation

- [Architecture](./docs/architecture.md)
- Design
  - [Business Logic Design](./docs/design-business-logic.md)
  - [LVA Topology Design](./docs/design-lva-topology.md)
  - [Integration Testing](./docs/design-integration-testing.md)
  - [DevOps Pipelines](./docs/devops-pipelines.md)
  - [Edge Layered Deployments](./docs/devops-layered-deployment.md)
- Development
  - [Environment Setup](./docs/dev-environment-setup.md)
  - [Edge Virtual Machine](./docs/dev-edge-virtual-machine.md)
  - [Troubleshooting](./docs/dev-iot-troubleshoot.md)

### Projects

- [Edge](./edge/README.md) - This project contains all Edge modules and deployment manifests
- [LVA Console App](./lva-console-app/README.md) - This project contains the console app that will run direct method operations on the
  LVA module running on the Edge device

### Prerequisites

- [Python v3.7](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/)
- [Visual Studio Code](https://code.visualstudio.com/)
- [Azure IoT Tools extension for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=vsciot-vscode.azure-iot-tools)

### Environment Setup

- [Follow python setup instructions and activate your virtual environment](./docs/dev-environment-setup.md)
- `pip install -r requirements.txt`

### Linting

See the lint section in the [environment setup document](./docs/dev-environment-setup.md)
