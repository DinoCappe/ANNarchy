# ANNarchy

Second-year project for Scuola Ortogonale.

## How to compile & run

First, create a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

After setting up Python, configure the C++ project (make sure the virtual environment is active):

```bash
cmake -S . -B app
```

Build the project:

```bash
cmake --build app -j <jobs_number>
```

At this point, everything should be ready. You can verify the installation by running:

```bash
python app/algorithm.py
```

## Useful links for nanobind
https://nanobind.readthedocs.io/en/latest/basics.html
https://nanobind.readthedocs.io/en/latest/exchanging.html 
