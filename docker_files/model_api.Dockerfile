# Start with a Python image as the base
FROM python:3.10

# Copy the MLflow model to the container
COPY ../Model_API/production_model /production_model

RUN pip install -r /production_model/requirements.txt

# Expose the default port for MLflow
EXPOSE 5000

# Serve the model
CMD ["mlflow", "models", "serve", "-m", "/production_model", "--env-manager", "local", "-h", "0.0.0.0", "-p", "5000"]