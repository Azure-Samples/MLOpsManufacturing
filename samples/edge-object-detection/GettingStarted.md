# Getting Started <!-- omit in toc -->

This section will walk through how to get started with your own edge object detection!

## Sections <!-- omit in toc -->

- [Prerequisites](#prerequisites)
- [DevOps Setup](#devops-setup)
- [Deployment/Resources Setup](#deploymentresources-setup)
- [Validating Deployments](#validating-deployments)

## Prerequisites

In order to deploy this project you will need the following:

1. Your own Azure Dev Ops instance to deploy everything to.

   [How to create a dev ops instance](https://docs.microsoft.com/en-us/azure/devops/organizations/accounts/create-organization?view=azure-devops)
1. An Azure subscription you can deploy resources to.

   [Sign up for a free Azure subscription](https://azure.microsoft.com/en-us/free/)
1. The [replaceTokens Azure Pipelines task](https://marketplace.visualstudio.com/items?itemName=qetza.replacetokens)

   Some of the `IoT Edge Modules` pipelines require an Azure DevOps task named `replaceTokens`
   that doesn't yet come default in Azure Dev Ops so it must be installed manually.

## DevOps Setup

In order to deploy this project and do object detection on the edge in your own resource group you will need to
[clone this repo into your own Azure Dev Ops instance](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-new-repo?view=azure-devops)
and take a few more steps.

You will need to do the following steps to get your Azure DevOps instance ready to deploy.

1. [Create an Azure service principal](https://docs.microsoft.com/en-us/cli/azure/create-an-azure-service-principal-azure-cli)

    ```bash
    az ad sp create-for-rbac --name <ServicePrincipalName>
    ```

    **Write down and remember it's ID and secret, you will need it later in these steps.**
1. In [vars-common.yml](./.pipelines/variables/vars-common.yml) update the values of:

   - `SUBSCRIPTION_ID` to be your [Azure Subscription ID](https://docs.microsoft.com/en-us/azure/media-services/latest/how-to-set-azure-subscription?tabs=portal)
   - `AAD_TENANT_ID` to be your [AAD Tenant ID](https://docs.microsoft.com/en-us/azure/active-directory/fundamentals/active-directory-how-to-find-tenant)
   - `AAD_SERVICE_PRINCIPAL_ID` to be the ID of the service principal you created
   - `projectName` to be 3-10 different letters. Perhaps letters in your name or your initials.

    Some of the resources need to have globally unique names and most of the names are created using
    `projectName`, `projectLocation`, and `suffix` (from vars-dev and vars-prod).
    Changing any of those should do the trick, but we are at the mercy of
    what resources others have already created to avoid naming conflicts.

    Unfortunately the names can't get too long though either, as storage accounts can only be 24 characters long.
1. Throughout this project many of our pipelines are set to run dev only off a certain branch,
    and prod off a different certain branch. For us those branches were `main` and `release`.
    If you want those branches to be different then do a find and replace of `refs/heads/<name>` to be whatever you want it to be.

    If your Azure DevOps default branch is `master`, then you will either need to change the conditions or
    [rename your default branch](https://dev.to/jessehouwing/rename-your-master-branch-in-azure-repos-3695) to `main`.
1. Create an [Azure Resource Manager service connection](https://docs.microsoft.com/en-us/azure/devops/pipelines/library/service-endpoints?view=azure-devops&tabs=yaml)
    from your Azure Dev Ops instance to whatever Azure subscription you want to deploy your resources to.
    This service connection should be at the subscription level, not to a specific resource group.
    Make sure "Grant access permission to all pipelines" is checked.

    You might want to create one dev service connection and one production one. Remember the names of both.
1. Update the pipeline files to use your newly created service connections.

    There are 3 different pipelines that use these service connections that you will need to update.
    For all of them update `<environment>ServiceConnection` variable in the `variables:` section of the file.
    The value you will set it to is whatever you named your service connection.

    - [CD IaC](./.pipelines/cd/iac.yml)
    - [CD IoT Edge Modules](./.pipelines/cd/iot-edge-modules.yml)
    - [CI IoT Edge Modules](./.pipelines/ci/iot-edge-modules.yml)

    Be sure to update both the dev and prod service connection values for all of them.
1. Now that you have service connections created you will need to find something called the objectId for those
    and update other variables.

    You will need to update the `servicePrincipalObjectId` value in both
    [vars-dev.yml](./.pipelines/variables/vars-dev.yml) and [vars-prod.yml](./.pipelines/variables/vars-prod.yml).
    The [devops-pipelines.md](./docs/devops-pipelines.md) has a section titled `Getting servicePrincipalObjectId`
    that will walk you through how to get that.
    > Note: This is for the Azure Dev Ops Resource Manager service connection created through Azure Dev Ops
    > not the one created through cli at the very beginning

    If you created a different dev and prod service connection the two values will be different.
1. Now you will need to create Azure Pipelines using the existing yaml files provided in this repo.
    If this is your first time creating a pipeline this link is slightly helpful
    [Create Azure Pipeline from existing YAML](https://stackoverflow.com/questions/59067096/create-a-new-pipeline-from-existing-yml-file-in-the-repository-azure-pipelines)

    There are 4 pipelines you need to create, and another 4 we recommend.
    We recommend naming all of them what we have,
    but we will call out any changes that need to be made if you name them differently

    For in depth documentation on all the pipelines, consult the [devops-pipelines.md](./docs/devops-pipelines.md) document

    Pipelines you need to create:
    1. `CD - IaC`

        This is the pipeline that will deploy all our Azure resources.
        It is from the [cd/iac.yml file](./.pipelines/cd/iac.yml)
    1. `CD - IoT Edge Modules`

        This is the pipeline that will deploy all the edge modules to iotedge.
        It is from the [cd/iot-edge-modules.yml file](./.pipelines/cd/iot-edge-modules.yml)
    1. `CI - IoT Edge Modules`

        This is the pipeline that builds and pushes the artifacts we will deploy to iotedge.
        It is from the [ci/iot-edge-modules.yml file](./.pipelines/ci/iot-edge-modules.yml).

        > **Important**
        >
        > If you choose to name this pipeline something other than `CI - IoT Edge Modules`
        > then you will need to update the `resources->pipelines->pipeline->source`
        > value in the [cd/iot-edge-modules.yml file](./.pipelines/cd/iot-edge-modules.yml) file.
        > That source must match the exact spelling of your pipeline name for deployments to automatically trigger.

        Leave the `resources->pipelines->pipeline` value the same,
        only change the `source` value if you opt to name your Azure Pipeline something other than `CI - IoT Edge Modules`.
    1. `CI - LVA Console App`

        This is the pipeline that builds and pushes the artifacts used to start LVA
        and trigger object detection and recordings once everything is deployed.
        It is from the [ci/lva-console-app.yml file](./.pipelines/ci/lva-console-app.yml).

        > **Important**
        >
        > If you choose to name this pipeline something other than `CI - LVA Console App`
        > then you will need to update the `resources->pipelines->pipeline->source`
        > value in the [cd/iot-edge-modules.yml file](./.pipelines/cd/iot-edge-modules.yml) file.
        > That source must match the exact spelling of your pipeline name for deployments to automatically trigger.

        Leave the `resources->pipelines->pipeline` value the same,
        only change the `source` value if you opt to name your Azure Pipeline something other than `CI - LVA Console App`.

    Pipelines we recommend creating:
    1. `CI - Docs`

        This is a pipeline that runs CI if documents are updated.
        It is from the [ci/docs.yml file](./.pipelines/ci/docs.yml)
    1. `PR - Docs`

        This is a pipeline that does PR validation if you are updating documentation.
        It is from the [pr/docs.yml file](./.pipelines/pr/docs.yml).
    1. `PR - IoT Edge Modules`

        This is a pipeline that does PR validation for changes to any iotedge modules.
        It is from the [pr/iot-edge-modules.yml file](./.pipelines/pr/iot-edge-modules.yml).
    1. `PR - LVA Console App`

        This is a pipeline that does PR validation for changes to LVA Console App.
        It is from the [pr/lva-console-app.yml file](./.pipelines/pr/lva-console-app.yml).

    All of the PR pipelines (if created) will not automatically trigger
    unless you add it as a build validation for a certain branch.
    We recommend adding all of them as build validation policies to protect your main branch.

    This link explains how to do that: [How to add Build Validation Policies](https://docs.microsoft.com/en-us/azure/devops/repos/git/branch-policies?view=azure-devops#build-validation)

    We recommend doing proper path filtering so they are only triggered for relevant file changes.

## Deployment/Resources Setup

Now that all the pipelines are created and the values are updated we can start deploying things!

The remaining steps left to take for deployment are:

1. Trigger the `CD - IaC` pipeline to run off the main branch. When this pipeline runs all of your Azure Resources will get created
    in the subscription you created the service connection for.
    They will all be deployed to the resource group according to how you named it in [vars-common.yml](./.pipelines/variables/vars-common.yml)
1. Now that the keyvault has been deployed you will need to add one more secret to it manually.

    In order to do this you will need to [add yourself to the keyvault access policies](https://docs.microsoft.com/en-us/azure/key-vault/general/assign-access-policy-portal).

    Now that you have permissions to view and create secrets, create a secret named `aadSpSecret` with the value of the
    service principal secret you created at the very first step. Told you you'd need to use it again :-).
1. Next, there are a couple of secrets from keyvault that are needed by CI and CD pipelines.
    To provide access to this we will be creating two variable groups, `secrets_dev` and `secrets_prod`, in Azure Pipelines.
    Both of these are needed to run the pipeline, so if you haven't deployed to a prod environment just create a placeholder `secrets_prod`.
    Another option is to delete the prod sections of the pipelines,
    but only do this if you know you won't want a prod environment in the future.

    These variable groups are created under Pipelines->Library and each of them will be linked to keyvault.
    Follow [these steps](https://docs.microsoft.com/en-us/azure/devops/pipelines/library/variable-groups?view=azure-devops&tabs=yaml#link-secrets-from-an-azure-key-vault)
    to link your secrets from keyvault.

    The specific secrets we will need to access from our pipelines are:
    - aadSpSecret
    - acrPassword
    - iotHubConnectionString
1. Now that the secret groups are created we can run our two necessary CI pipelines.
    Trigger `CI - LVA Console App` first and then `CI - IoT Edge Modules` to run off the main branch.

    Once these have both completed the `CD - IoT Edge Modules` will be automatically triggered
    and your modules will be deployed and object detection on the edge will have started!!!

## Validating Deployments

To see the objects that are detected you can look at the `media service` resource `assets` in the Azure Portal.
This is where video recordings will be uploaded. You can also connect to the VM using Bastion
and look at the `objectDetectionBusinessLogic` logs (command to run is `iotedge logs objectDetectionBusinessLogic`).
This will show you what objects are getting detected in your video.

To stop the object detection you can shut down the VM or follow our [environment setup instructions](./docs/dev-environment-setup.md)
and the documentation in [LVA Console App](lva-console-app/README.md) and run the console app with the `operations_teardown.json` file :-).
This is the option we prefer of course :-), but both would work.

One last note:

We recommend adding a manual approval gate for prod deployments,
so that the prod environment can only be pushed to after someone with authority approves.
[This link](https://docs.microsoft.com/en-us/azure/devops/pipelines/process/approvals?view=azure-devops&tabs=check-pass)
will explain how to do that.

