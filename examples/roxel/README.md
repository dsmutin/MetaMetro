# Roxel street CFA

Builds a de Bruijn CFA from the built-in sfnetworks example:

```r
gr <- sfnetworks::as_sfnetwork(sfnetworks::roxel)
```

Nodes are the street-network vertices. A directed edge is one street segment. Sequences are filled from one base-4 counter tape so neighboring nodes share a repeat of length `k - 1`. Every character attribute on the edges becomes a colour namespace, and each node and edge carries the values it has.

Requires R with the `sf`, `sfnetworks`, and `tidygraph` packages.

```bash
python examples/roxel/build_cfa.py --out data/work/roxel/cfa
```

The script checks every forward overlap and compacts the CFA. A mismatch exits non-zero and writes nothing. The sequences are a simulated tape, not an observed genome.
