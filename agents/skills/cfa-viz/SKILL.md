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
- [ ] Stack facet panels vertically
- [ ] Omit positions for Fruchterman–Reingold, or pass longitude and latitude for every node
- [ ] Write PDF or SVG with axis units and legends outside the panels
```

## Call

```python
from metametro.viz.cfa_colouring import ColourFacet, ColourLabel, plot_cfa_colouring

plot_cfa_colouring(
    graph,
    node_label=ColourLabel(
        namespace="coverage",
        legend="Simulated passengers (sum)",
        column="coverage",
        palette="pink_yellow_green",
    ),
    edge_label=ColourLabel(
        namespace="coverage",
        legend="Simulated passengers (mean)",
        column="coverage",
        palette="pink_yellow_green",
    ),
    facet=ColourFacet(
        namespace="transport_type",
        levels=("bus", "tram", "trolley"),
        level_labels={"bus": "Bus", "tram": "Tram", "trolley": "Trolleybus"},
    ),
    path="colouring.pdf",
    x_label="Layout x",
    y_label="Layout y",
    layout_seed=0,
    layout_spread=spread,
    facet_along="y",
)
```

`facet_along="y"` stacks the panels. Omit `positions` for Fruchterman–Reingold. `layout_seed` fixes the start. The ideal spacing is `layout_spread * sqrt(1 / n)`. `nearest_neighbor_ratio` divides the median nearest-neighbor distance by that spacing. A ratio near 1 is the equilibrium spacing. A ratio much below 1 means the layout stopped early. The axes fit the drawing, so a larger `layout_spread` rescales it and does not unpack a clump. Axis labels are dimensionless layout coordinates.

To draw a map, pass a longitude and latitude for every node and set the axis labels to `Longitude (°E)` and `Latitude (°N)`. A missing coordinate is an error. Do not drop that node and do not invent a coordinate.

`cfa_colour_frames` returns the same numbers without drawing. Every facet level maps every `node_id` and every `edge_id`. `None` is NA.

## Labels

| Input | Meaning |
| --- | --- |
| `node_label.namespace` | Which `colors.tsv` rows colour the nodes |
| `edge_label.namespace` | Which rows colour the edges |
| `matches(level, value)` | Which of those values belong to this facet level |
| `reduce(values)` | One number or string. `None` or an empty numeric NaN becomes NA |
| `column` | Numeric node or edge column. Used when the element is in the facet; otherwise NA |
| `palette` | `YlGnBu` by default; `pink_yellow_green` for pink → yellow → light green; `Set1` for text |
| `legend` | Colourbar or legend title, including the unit |
| `facet.namespace` | Panel split. Membership is "this colour id is on the element" |

An element without the facet colour is NA even when `reduce([])` would be `0`. `count_colours([])` is `0` only for members.

Text palette `Set1`: `Other` is last and gray80. `Unclassified` is gray20. Do not invent a third grey.

## Figure

- PDF or SVG only. No figure title. Facet captions are panel labels.
- Legends and colourbars sit outside the panels.
- Axis labels name the quantity and the unit.
- This repository has no ggplot `theme_main`. The default numeric palette is `YlGnBu`. Text uses `Set1`. NA is gray80. Legends stay outside.
- The usual figure omits `positions` and uses Fruchterman–Reingold (`layout_seed`, `layout_spread`).
- Facet panels are stacked vertically (`facet_along="y"`).
- `pink_yellow_green` is pink (`#FF9EC7`), yellow (`#FFE56A`), then light green (`#B6F2A0`). Use it on both the node scale and the edge scale when the caller asks for that gradient.

## Saint Petersburg example

```bash
python examples/spb_transit/plot_colouring.py \
  --cfa data/work/spb_ground_transit/cfa \
  --out data/work/spb_ground_transit/cfa_colouring.pdf
```

A node is one named stop: feed ids with the same name at most 60 m apart. The PDF stacks Bus, Tram, and Trolleybus vertically. Both pages use the pink–yellow–light-green gradient. Node colour is simulated passengers: each route draws Normal(mean = its stop count, sd = sqrt of that count) with seed 0, and the node sums the routes that serve it. Edge colour is the mean of the two stops. Page 1 is longitude and latitude. Page 2 is the Fruchterman–Reingold spread whose median nearest-neighbor distance is closest to the ideal spacing. The feed has no measured passenger counts.
