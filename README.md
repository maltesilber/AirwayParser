# AirwayParser

AirwayParser extracts the centerline tree of a 3D airway segmentation. Implementation of the paper Minimum-Cost Path Framework for 3D Airway Centerline Extraction, presented at MICCAI 2026 Workshop on Thoracic Image Analysis.

`parse_tree` traces a sub-voxel minimum-cost path from every terminal branch back to the trachea and merges the paths into a tree of branches with parent/child relations and generation numbers.

## Usage

```python
from parser import parse_tree

# mask: 3D binary numpy array with the trachea at the high end of the last axis
# spacing: voxel size in mm, same axis order as the mask
tree = parse_tree(mask, spacing=(0.6, 0.6, 1.0))

for b in tree.branches.values():
    print(b.id, b.parent, b.generation, len(b.points))
```

Returns a `BranchTree`: `branches` maps id to `Branch`, `root()`, `leaves()` and `branch_points()` return the root branch, the terminal branches and the bifurcation points. A `Branch` has `id`, `parent`, `generation`, `children`, `points` (an `(n, 3)` centerline in voxel coordinates, root end first), `start_point`, `end_point` and `is_leaf`.

## Installation
Clone the repository and install the requirements:

```bash
pip install -r requirements.txt
```

Python >= 3.10. Run from the repository folder or put it on `PYTHONPATH`.



## License

MIT
