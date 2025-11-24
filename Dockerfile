# ###########################################################################
#
# @author      James R. Heselden (github: iranaphor)
# @maintainer  James R. Heselden (github: iranaphor)
# @datecreated 24th November 2025
# @credits     Code structure and implementation were developed by the
#              author with assistance from OpenAI's GPT-5.1 model, used
#              under the author's direction and supervision.
#
# ###########################################################################

# Use a slim Python base image
FROM python:3.11-slim

# Install system dependencies (optional but often useful for scientific Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
# yaml  -> PyYAML
# numpy -> numpy
# matplotlib -> matplotlib
# fastkml -> fastkml
# shapely -> shapely
RUN pip install --no-cache-dir \
    pyyaml \
    numpy \
    matplotlib \
    fastkml \
    shapely

# Copy your project into the image (adjust as needed)
COPY ./app /app

# Set the default DATA_PATH inside the container
ENV DATA_PATH=/data

# By default, run python3 (you can override the script in docker-compose)
CMD ["python3", "/app/rf_polar_maps.py"]
#CMD ["python3", "/app/test_perms.py"]
