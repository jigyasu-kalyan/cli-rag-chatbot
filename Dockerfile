# Start with an official Python base image (use a specific version)
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install Python dependencies
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
# This includes main.py, rag_logic.py, .env, and the data/ folder
COPY . .

# Make port 8000 available outside the container (FastAPI default)
EXPOSE 8000

# Define the command to run your application when the container starts
# Use 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]