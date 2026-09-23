# Saint Petersburg ground-transit CFA

Builds a de Bruijn CFA from the public ORGP GTFS feed
(https://transport.orgp.spb.ru/Portal/transport/internalapi/gtfs/feed.zip).

Nodes are stops. A directed edge is a pair of stops that follow each other on a trip. Sequences are filled from one base-4 counter tape so neighboring nodes share a repeat of length `k - 1`. Each stop in this feed belongs to one mode, so a mode count would be 1 on every node. Node coverage is the mean number of routes over bus, trolley, and tram, with a missing mode counted as 0. Edge coverage is the mean of the two stops' coverages. That number is not a passenger count; the feed does not publish one.

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
  --gtfs /path/to/feed.zip \
  --out data/work/spb_ground_transit/cfa_colouring.pdf
```

Three panels: Bus, Tram, Trolleybus. Node colour is the percent of that mode's routes with an even number. Edge colour is the number of that mode's routes on the hop. A stop or hop that is not in the panel's mode stays on the map in NA grey. Positions are the GTFS coordinates. The feed has no passenger counts, and this figure does not invent them.
