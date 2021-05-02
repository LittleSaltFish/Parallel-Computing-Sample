# usage

1. write web relation to `Relation-M.csv` , strat from 0
   - or use `rand.py` to generate the file
   - execute `python3 rand.py {sizeof web nodes}`
2. execute `python3 trans.py {size of web nodes} {out percentage}`
3. execute `mpiexec -n {p number} python3 PageRanke.py {number of web nodes}`
