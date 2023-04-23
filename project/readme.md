# Projet - Eternity II

Eternity II is a famous puzzle asking to fill a 16 ×16 size board with 256 squares. Each square is divided into 4 areas containing a different color for each edge. A square can only be adjacent to another square with the same color on the adjacent edge.



## Run the project
→ Start a virtual env.
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

→ Run problem
```
cd code/
python3 main.py --agent=random --infile=instances/eternity_A.txt
python3 main.py --agent=heuristic --infile=instances/eternity_A.txt
python3 main.py --agent=local_search --infile=instances/eternity_A.txt
python3 main.py --agent=advanced --infile=instances/eternity_A.txt
```