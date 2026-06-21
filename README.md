# ANNarchy

Second-year project for Scuola Ortogonale — approximate nearest neighbor search in C++ with a Python wrapper via nanobind.

## Running with Docker (submission format)

### Prerequisites
- Docker installed
- A dataset `.hdf5` file downloaded locally

### Step 1 — Build the base image

The competition base image is private, so everyone builds it locally once:

```bash
curl -o harness.py https://raw.githubusercontent.com/Cecca/orthogonal-competition/main/harness.py

cat > Dockerfile.base << 'EOF'
FROM python:3.11-slim
RUN pip install --no-cache-dir numpy h5py pyyaml
COPY harness.py /app/harness.py
ENTRYPOINT ["python", "/app/harness.py"]
EOF

docker build -f Dockerfile.base -t ann-orthogonal/base:latest .
```

### Step 2 — Build the submission image

```bash
git clone https://github.com/DinoCappe/ANNarchy.git
cd ANNarchy
git checkout first-submission
docker build -t annarchy/nns:latest .
```

This takes a few minutes — it compiles the C++ extension inside the container.

### Step 3 — Test with the harness

```bash
docker run --rm \
  -v /path/to/your/dataset.hdf5:/data/dataset.hdf5 \
  -e DATASET_PATH=/data/dataset.hdf5 \
  -e DATASET_NAME=yahoo-minilm-public \
  -e SCENARIO_NAME=high_recall \
  -e RESULTS_PATH=/tmp/results.hdf5 \
  -e QUERY_K=100 \
  annarchy/nns:latest
```

Replace `/path/to/your/dataset.hdf5` and `DATASET_NAME` with the dataset you are using.
`SCENARIO_NAME` can be `high_recall`, `fast`, or `memory`.

A successful run prints something like:
```
[harness] fit() done: 85.6s memory_mb=980.1
[harness] queries done: total=119.5s  QPS=8.4  median=110ms  n_dist_queries=169335425
[harness] Results written to /tmp/results.hdf5
```

---

## Local development

### Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Build the C++ extension

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

### Smoke test

```bash
python app/algorithm.py
```

### Benchmark

```bash
python benchmark.py datasets/your-dataset.hdf5
```

Measures recall, build time, query time, QPS, memory, and distance count across all three scenarios (`high_recall`, `fast`, `memory`).
