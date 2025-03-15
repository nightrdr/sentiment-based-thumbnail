# Use the gigante/python-opencv base image
FROM python:3.8.12

# Set the working directory
WORKDIR /app

# -----------------------------------
# update image os
# Install system dependencies
RUN apt-get update && apt-get install -y \
  ffmpeg \
  libsm6 \
  libxext6 \
  libhdf5-dev \
  && rm -rf /var/lib/apt/lists/*

# Install FastAPI and Uvicorn
RUN pip install --no-cache-dir fastapi uvicorn

# Copy the requirements file and install dependencies
COPY . .
# install dependencies
RUN pip install -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
