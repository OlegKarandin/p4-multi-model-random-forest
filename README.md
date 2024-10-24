# Implementation of Multiple Concurrent Tree-Based Models in P4 Switches using Feature Sharing

Please, check if the two dataset, "Wednesday-workingHours.pcap_ISCX.csv" and "apps_flow_features.csv", are automatically cloned on your machine with the proper data inside, otherwise download them manually.

## Environment Setup

### python
Python version 3.11.x (highly recommended)

### conda
If you use conda, create the environment as follows (subsitute <env> with the environment name):

`
conda env create -f environment.yml
`

### pip
If you use pip, create the environment as follows (subsitute <env> with the environment name):

```
python3 -m venv <env>
source <env>/bin/activate
pip install -r requirements.txt
```

## Generating P4 code
To generate the P4 that implements Random Forest models for DDOS detection and Application Identification code run
`
python main.py
`

Resulting code and M/A table entries will be generated in ./P4 folder

## Modifying the number of trees
Number of trees (and thus size of M/A tables) can be configured in `main.py`:
```
num_trees_app = 3
num_trees_ddos = 1
```
