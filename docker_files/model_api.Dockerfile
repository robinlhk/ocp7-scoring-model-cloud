# Start with a Python image as the base
FROM python:3.10

# Copy the MLflow model to the container
COPY ../notebooks/custom_model /custom_model

RUN pip install -r /custom_model/requirements.txt

# Expose the default port for MLflow
EXPOSE 5000

# Serve the model
CMD ["mlflow", "models", "serve", "-m", "/custom_model", "--env-manager", "local", "-h", "0.0.0.0", "-p", "5000"]