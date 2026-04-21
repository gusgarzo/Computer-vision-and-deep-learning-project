# EDA Summary

- `Road_poles_iPhone`: 942 train / 261 val
- `roadpoles_v1`: 322 train / 92 val
- Median box area: 0.078% vs 0.061%
- Mean height/width ratio: 4.28 vs 16.32
- Mean boxes/image: 1.24 vs 1.22

## Presentation takeaway

- The datasets should not be presented as identical tasks.
- `roadpoles_v1` is the harder benchmark because poles are visually thinner and smaller.
- Both datasets support the same modeling intuition: high resolution is more important than handling dense clutter.