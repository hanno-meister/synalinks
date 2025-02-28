# Enabling LM Tracing

Tracing is important for several reasons, especially in the context of machine learning and software development. It helps identify issues and bugs in your Synalinks programs by providing a detailed log of events and operations. This makes it easier to pinpoint where things went wrong and why.

To install MLflow, an open-source tracing software, use the following:

```shell
pip install mlflow
```

To activate the LM tracing, add the following lines to the top of your script

```python
import mlflow

mlflow.litellm.autolog()
# Set this to your MLflow server
mlflow.set_tracking_uri("http://localhost:5000")
```

You are done, MLflow is now configured.

To launch MLflow server, use the following command in a shell

```shell
mlflow server --host 127.0.0.1 --port 5000
```

You can find more information about the server configuration [here](https://mlflow.org/docs/latest/tracking/server.html).