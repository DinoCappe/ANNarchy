# -----------------------------------------------------------------------
# NNS Competition – Student Dockerfile
#
# Build and push:
#   docker build -t <your-team>/nns:latest .
# -----------------------------------------------------------------------

FROM ann-orthogonal/base:latest

# Install cmake (base image already has build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*

# Install nanobind (needed to compile the C++ extension)
RUN pip install --no-cache-dir nanobind

# Build the C++ extension and install it into site-packages so that
# `import ANNarchy` works regardless of the working directory
COPY ANNarchy.cpp   /build/ANNarchy.cpp
COPY CMakeLists.txt /build/CMakeLists.txt
RUN cmake -S /build -B /build/out \
    && cmake --build /build/out -j$(nproc) \
    && find /build/out -name "ANNarchy*.so" \
         -exec cp {} $(python3 -c "import site; print(site.getsitepackages()[0])") \;

# Copy algorithm and scenario definitions
COPY app/algorithm.py   /app/algorithm.py
COPY app/scenarios.yaml /app/scenarios.yaml
