# Saint Petersburg ground-transit CFA

Builds a de Bruijn CFA from the public ORGP GTFS feed
(https://transport.orgp.spb.ru/Portal/transport/internalapi/gtfs/feed.zip).

A node is one named stop. Feed ids that share a name and lie within 60 m are merged; that is where the same-name cross-mode distance histogram stops falling. A directed edge is a pair of those stops that follow each other on a trip. Sequences are filled from one base-4 counter tape so neighboring nodes share a repeat of length `k - 1`.

Coverage is simulated, not taken from the feed. Each route draws one value from a normal distribution with mean equal to its number of stops and standard deviation equal to the square root of that number (seed 0). Negative draws are set to 0. The value is added at every stop the route serves, so a node is the sum of its routes. An edge is the mean of its two stops. The feed does not publish passenger counts.

Colours use two namespaces: `transport_type` (`bus`, `trolley`, `tram`) and `route` (`bus:193`, and the route id as well when that label is shared). Each stop and each hop carries every colour that applies.

## Run

```bash
python examples/spb_transit/build_cfa.py --gtfs /path/to/feed.zip --out data/work/spb_ground_transit/cfa
```

The script validates the CFA, checks every forward overlap, and compacts it to a CDBG. A mismatch exits non-zero and writes nothing.

## Colouring figure

```bash
python examples/spb_transit/plot_colouring.py \
  --cfa data/work/spb_ground_transit/cfa \
  --out data/work/spb_ground_transit/cfa_colouring.pdf
```

The PDF stacks Bus, Tram, and Trolleybus vertically. Page 1 uses the longitude and latitude stored on the nodes. Page 2 uses the Fruchterman–Reingold spread whose median nearest-neighbor distance is closest to the ideal spacing. Both pages colour nodes by the simulated passenger sum and edges by the mean of the two stops, with the pink–yellow–light-green gradient. A stop or hop outside the panel's mode stays drawn in NA grey.
