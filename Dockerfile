# Use the Nvidia CUDA base image
FROM nvidia/cuda:12.5.0-base-ubuntu22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package lists
RUN apt-get update && apt-get upgrade -y

# Install Python and pip
RUN apt-get install -y python3 python3-pip

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy requiremehnts.txt and install Python packages
COPY requirements.txt /workspace/requirements.txt

# Install additional necessary packages
RUN pip3 install -r /workspace/requirements.txt

# Set the working directory
WORKDIR /workspace

# Expose Jupyter Notebook port
EXPOSE 8888

# Set the default command to run Jupyter Notebook
CMD [ "jupyter", "notebook", "--no-browser", "--allow-root", "--ip=0.0.0.0" ] 