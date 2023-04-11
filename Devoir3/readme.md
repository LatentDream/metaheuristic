
## Start a python virtual environment
`Python 3.10` was used
``` Shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Run the variable neighborhood search

/!\ Current search time is set at 20m 

→ To change this: [Line 15](solver_VND.py) (The algo is able to quickly generate a valid solution, so the value can be set to 1 second)

``` Shell
python3 main.py --agent=advanced --infile=instances/instance_A_30.txt
```

