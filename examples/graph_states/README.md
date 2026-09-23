# Geometric states

`examples/graph_states/build.py` compacts a CFA into a ToCUMG (CDBG) and a coloured graph tensor whose node features include mean longitude and latitude.

The PDF has three pages, in order: CFA, ToCUMG, CGT. Pass `layout="fr"` for Fruchterman–Reingold coordinates (`Layout x`, `Layout y`) instead of longitude and latitude. A graph with a `coverage` column uses that column and one shared YlGnBu scale. Otherwise edges are coloured by `--draw-edge-namespace` (Set1) and nodes by degree.

The GCN reads only longitude and latitude. It predicts one node class (softmax) and an independent 0/1 probability for every edge colour in `--edge-namespaces` (sigmoid, threshold 0.5). Metrics in `gcn/metrics.json` are computed on the seed-0 split.
