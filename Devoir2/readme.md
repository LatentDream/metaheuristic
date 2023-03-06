
# Assignment 2 - Home Care Optimization

Let $G = (N, A)$ be a complete, directed graph with $N = {0,1,...,n}$. $N$ is the set of nodes, node 0 represents the starting node and $A$ is an $N × N$ matrix where $a_{i j} ∈ A$ represents the time needed to get to node (patient) $i$ from node (patient) $j$. Moreover, for each patient, there exists a time window $[l_i ,u_i ]$ where patient $i ∈ N$ is available. This time window indicates that patient $i$ cannot be visited before $l_i$ or after $u_i$ . We admit a waiting time, i.e., it is possible to arrive at a patient before the start of his availability as long as we wait until he is available. It is not possible to leave a patient $i$ before the beginning of his availability $l_i$ . It is also assumed that the time to treat a patient (i.e. the time the doctor stays at the patient's home) is zero.


Thus, given a tour $P$, the departure time $D_{p_k}$ of patient $p_k$ is $D_{p_k} = max(A_{p_k} ,l_{p_k})$ where $A_{p_k} = D_{p_{k-1}} + a_{p_{k-1}p_k}$ . 

In other words, the departure from a patient $p_k$ is the maximum value between: 
1. the lower bound of its availability $(l_{p_k} )$ and
2. the departure value of the previous patient in the round $(D_{p_{k-1}})$ + the travel time to reach the current patient $(a_{p_{k-1}p_k} )$. 

A solution to the problem is represented by $P = (p_0 = 0, p_1,...,p_n,p_{n+1} = 0)$ where $(p_1,p_2,...,p_n)$ is a permutation of nodes of $N$ In other words, it is a Hamiltonian cycle where $p_k$ is the index of the patient visited in the $k^{th}$ position of the tour. 

Mathematically, we have the following model:

$$ \textrm{minimize } D_{p_{n+1}} $$
$$ \textrm{Subject to } \sum_{k=o}^{n+1} \omega\left(p_k \right) = 0$$

Where 

$$  \omega \left(p_k \right) = 1  \textrm{ if } A_{p_k} > u_{p_k} \textrm{ else } 0$$
$$A_{p_{k+1}} = \textrm{max}(A_{p_{k}}, l_{p_k}) + a_{p_{k}p_{k+1}}$$

The constraint $\sum_{k=o}^{n+1} \omega\left(p_k \right) = 0$ ensures that all constraints on the time windows are satisfied.
k=0
Note that a valid solution must 1. visit all patients one and only one time, and 2. respect the
time window constraints.

---
→ Start a env
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
→ Run problem
```
python3 main.py --agent=naive --infile=instances/A_4.txt
python3 main.py --agent=advanced --infile=instances/B_20.txt
```
---
Solution tested:
1. Beam-ACO for the travelling salesman problem with time windows from Manuel Lopez-Ibanez and Christian Blum

Beam-ACO algorithms are hybrids between ant colony optimization and
beam search. Ant colony optimization (ACO) is a metaheuristic that is based on the probabilistic construction of solutions. At each algorithm iteration, a number of solutions are constructed independently of each other. Beam-ACO employs instead at each iteration a probabilistic beam search procedure that constructs a number of solutions interdependently and in parallel. At each construction step, beam search keeps a certain number of the best partial solutions available for further extension. These partial solutions are selected with respect to bounding information. Hence, accurate and inexpensive bounding information is a crucial component of beam search. A problem arises when the bounding information is either misleading or when this information is computationally expensive, which is the case for the TSPTW. This work uses stochastic sampling as an alternative to bounding information. When using stochastic sampling, each partial solution is completed a certain number of times in a stochastic way. The information obtained by these stochastic samples is used to rank the different partial solutions. The worst partial solutions are then excluded from further examination. 