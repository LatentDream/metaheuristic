# Optimization of a telecommunication network

## Goal

Minimize construction costs and penalties associated with the exclusion of certain regions (network end nodes)

- Implementation of a local search with a mechanism to extract local minimums

## Run the porject

→ Start a env
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
→ Run problem
```
python3 main.py --agent=hernes --infile=instances/reseau_B_64_192_64.txt
python3 main.py --agent=advanced --infile=instances/reseau_B_64_192_64.txt
```
