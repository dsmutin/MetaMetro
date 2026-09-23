---
name: cfa-viz
description: >-
  Plot a faceted CFA colouring as a PDF or SVG. Node colour, edge colour,
  and palettes are chosen from colours.tsv labels; elements outside a facet
  stay visible with NA colour. Use when the user asks to visualize a CFA,
  colour a graph, facet a colouring, or invokes CFA-viz.
disable-model-invocation: true
---

# CFA-viz

Draw a CFA colouring. Do not drop nodes or edges that lack the facet label. Paint them with NA grey (`#CCCCCC`, gray80).

## Checklist

```
CFA-viz:
- [ ] Load the CFA with load_cfa (validation on)
- [ ] Choose node label, edge label, palettes, and facet
- [ ] Resolve frames and check non-members are None, not absent
- [ ] Pass a position for every node, or omit positions for a circle
- [ ] Write PDF or SVG with axis units and legends outside the panels
```

## Call

```python
from metametro.viz.cfa_colouring import (
    ColourFacet,
    ColourLabel,
    count_colours,
    even_route_percent,
    plot_cfa_colouring,
)

plot_cfa_colouring(
    graph,
    node_label=ColourLabel(
        namespace="route",
        legend="Even-numbered routes (%)",
        matches=lambda level, value: value.startswith(f"{level}:"),
        reduce=even_route_percent,
        palette="YlGnBu",
        limits=(0.0, 100.0),
    ),
    edge_label=ColourLabel(
        namespace="route",
        legend="Routes (count)",
        matches=lambda level, value: value.startswith(f"{level}:"),
        reduce=count_colours,
        palette="YlGnBu",
    ),
    facet=ColourFacet(
        namespace="transport_type",
        levels=("bus", "tram", "trolley"),
        level_labels={"bus": "Bus", "tram": "Tram", "trolley": "Trolleybus"},
    ),
    positions=positions,
    path="colouring.pdf",
    x_label="Longitude (°E)",
    y_label="Latitude (°N)",
    aspect=aspect,
)
```

`cfa_colour_frames` returns the same numbers without drawing. Every facet level maps every `node_id` and every `edge_id`. `None` is NA.

## Labels

| Input | Meaning |
| --- | --- |
| `node_label.namespace` | Which `colors.tsv` rows colour the nodes |
| `edge_label.namespace` | Which rows colour the edges |
| `matches(level, value)` | Which of those values belong to this facet level |
| `reduce(values)` | One number or string. `None` or an empty numeric NaN becomes NA |
| `palette` | `YlGnBu` for numbers, `Set1` for text |
| `legend` | Colourbar or legend title, including the unit |
| `facet.namespace` | Panel split. Membership is "this colour id is on the element" |

An element without the facet colour is NA even when `reduce([])` would be `0`. `count_colours([])` is `0` only for members.

Text palette `Set1`: `Other` is last and gray80. `Unclassified` is gray20. Do not invent a third grey.

## Figure

- PDF or SVG only. No figure title. Facet captions are panel labels.
- Legends and colourbars sit outside the panels.
- Axis labels name the quantity and the unit.
- This repository has no ggplot `theme_main`. Follow the same choices: `YlGnBu`, `Set1`, NA gray80, outside legends.
- Missing coordinates are an error. Do not drop those nodes and do not invent positions.
- Circle layout is only the fallback when the caller omits `positions`.

## Saint Petersburg example

```bash
python examples/spb_transit/plot_colouring.py \
  --cfa data/work/spb_ground_transit/cfa \
  --gtfs /path/to/feed.zip \
  --out data/work/spb_ground_transit/cfa_colouring.pdf
```

Node colour: percent of that mode's routes with an even leading number (`145Б` is 145). Edge colour: count of that mode's routes. Panels: Bus, Tram, Trolleybus. The GTFS feed has no passenger counts.
