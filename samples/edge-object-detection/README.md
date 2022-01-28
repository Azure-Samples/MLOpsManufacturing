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

As we enter a world with more and more connectivity in all aspects of our lives
more opportunities to leverage technology arise.
One such use case that has seen a rapid rise in usage recently is running
machine learning on edge devices.
This particular sample shows how to set up object detection on edge devices
and propagate that detection up to the cloud, where from there
many different alerts can be set up.

Some example use cases of this are safety scenarios at factories without much foot traffic:

- Make sure no boxes are placed in a restricted danger zone
- Make sure there are no fires or spills
- Make sure no machines are jammed

Other use cases could be to enforce rules, e.g. make sure only EV cars park in EV parking spaces.

Lastly, some use cases can be more experimental and fun, e.g. trigger a notification
any time a dog walks past your window :-).

This sample lays out the infrastructure and pipelines for implementing all of those solutions
and gives a full running implementation of detecting when a truck drives past your window.
(But of course you can easily tweak that to be to detect whatever use case you want)

## Solution Summary

This sample solves this problem by utilizing Azure Video Analytics (AVA).
At the time of creating this sample it was named Live Video Analytics (LVA), so any references to LVA refer to that.
This solution takes in camera input and sends messages through IoT Hub
to trigger any cloud based alerting functionality you want.

This sample also sets up IaC pipelines for deploying all the needed resources,
and CI/CD pipelines for the different components.

The main component that will change with each use case is the python `objectDetectionBusinessLogic`
which is responsible for choosing under which conditions certain object
detections will trigger messages to the cloud.

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
