# AirwayParser

AirwayParser extracts the centerline tree of a 3D airway segmentation and computes centerline-based segmentation metrics. Implementation of the paper Minimum-Cost Path Framework for 3D Airway Centerline Extraction, presented at MICCAI 2026 Workshop on Thoracic Image Analysis.

<p align="center">
<!-- drop docs/pipeline.png here -->
</p>

# Installation
Install the package using pip:

```bash
pip install airwayparser
```
Python >= 3.10. For development, clone the repository and `pip install -e .`.

# Usage

### Skeletonizing

```python
from airwayparser import parse_tree

# mask: 3D binary numpy array in order (x, y, z) with the trachea at the high end of the last axis
# spacing: voxel size in mm, same axis order as the mask
tree = parse_tree(mask, spacing=(0.6, 0.6, 1.0))

for b in tree.branches.values():
    print(b.id, b.parent, b.generation, len(b.points))
```

`parse_tree` traces a minimum-cost path from every terminal branch back to the trachea and merges the paths into a tree of branches with parent/child relations and generation numbers.

| Backtracing | Branch point detection |
|:-:|:-:|
| |  |

### Segmentation metrics

```python
from airwayparser import evaluate

evaluate(prediction, reference, spacing)  
# {'DSC': ..., 'Precision': ..., 'clDice': ..., 'TLD': ..., 'BD': ...}, both trees parsed inside
```

## License

MIT
