FROM mcr.microsoft.com/azureml/inference-base-2004@sha256:5ec5af5ae2fea6626658ca8856d01174b001e8f00d5a763d625fc71f9794c3d8

WORKDIR /
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/minimal
ENV AZUREML_CONDA_DEFAULT_ENVIRONMENT=$AZUREML_CONDA_ENVIRONMENT_PATH

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

# Create conda environment
USER root
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

# install the azureml packages
RUN pip install 'azureml-automl-dnn-vision==1.52.0.post1' \
                'azureml-inference-server-http==0.8.4' \
                'azureml-contrib-services==1.52.0' \
                'azureml-automl-core==1.52.0.post1'

USER dockeruser

CMD [ "runsvdir", "/var/runit" ]