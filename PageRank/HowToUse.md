# usage

1. make `Relation-M.csv`
   1. write web relation to `Relation-M.csv` , remember nodes id strat from 0
   2. use `rand.py` to generate the file：execute `python3 rand.py {sizeof web nodes}`
2. execute `python3 trans.py {size of web nodes} {out percentage}`
3. execute `mpiexec -n {p number} python3 PageRanke.py {number of web nodes}`

# meaning

- `Relation-M.csv`  Relation of each nodes eg:0,1 means node0->node1
- `Adjacency-M.csv`  Adjacency Matrix of each nodes
- `Power-iterations-M.csv`  Matrix to multiply each round
- `Value-M.csv`  Final result

# Iteration times

To modify Iteration times,change `for i in range(100):`

default is 100

# About methods

In `trans.py`:

1. `method0` means no modify to node relation
2. `method1` means define each node link to itself
3. `method2` means if all 0 cols, transfer it to `1/n`
