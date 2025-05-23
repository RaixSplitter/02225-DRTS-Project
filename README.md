# 02225-Distributed-Real-Time-Systems Project

## Overview

This repository contains the project for the course 02225 Distributed Real-Time Systems at the Technical University of Denmark. 

## Config files

The configuration files are located in the `conf` folder. The configuration files are used to configure the main run of an experiment and is handled by Hydra. The configuration files are written in YAML.

For more details on configuration, see the [README in the `conf` folder](conf/README.md).

## Usage

### Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/RaixSplitter/02225-DRTS-Project.git
    ```

2. Create a virtual environment and install dependencies:

    ```bash
    cd 02225-DRTS-Project
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .
    ```

### Running demos with Makefile

For convenience, a Makefile is provided to run the demo. The demo can be run with the following command:

```bash
make run
```

To specify a different config file, use the `CONFIG` variable:

```bash
make run CONFIG=<config_name>
```

i. e., which is the default call
```bash
make run CONFIG=experiment
```

### Running demos with Python

To run an experiment with Python, run the following command:

```bash
python simsystem/run.py
```

Notice that one can specify it's own config file by using the `--config-name` flag, in which case the program will look for the config file in the `Exercises/conf` folder. For example:

```bash
python simsystem/run.py --config-name <config_name>
```

The run script scans the specified tasks folder for all csv files and runs an experiment on all tasks. The results are saved in the specified outputs folder. As per default the results are saved in the `results` folder.

Notice that each task file gets it's own output folder.

### Logging

Log files can be found in the `outputs` folder and are generated and managed by Hydra.
The recommended log is the report.log, which does not contain the debug information during development.