# AirwayParser

AirwayParser extracts the centerline tree of a 3D airway segmentation and computes centerline-based segmentation metrics. Implementation of the paper Minimum-Cost Path Framework for 3D Airway Centerline Extraction, presented at MICCAI 2026 Workshop on Thoracic Image Analysis.

<p align="center">
<img width="2400" height="793" alt="pipeline" src="https://github.com/user-attachments/assets/11b408df-a44d-4020-b40d-782ac392ca95" />
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
<div align="center">
    
| Backtracing | Branch point detection |
|:-:|:-:|
| <img width="340" height="400" alt="backtracing" src="https://github.com/user-attachments/assets/7d54db08-3228-47f7-8e4e-689e4f67d341" /> | <img width="340" height="400" alt="branch_detection" src="https://github.com/user-attachments/assets/233a6edc-79ed-4070-8ea2-9c283d0cd26b" /> |

</div>

### Segmentation metrics

```python
from airwayparser import evaluate

evaluate(prediction, reference, spacing)  
# {'DSC': ..., 'Precision': ..., 'clDice': ..., 'TLD': ..., 'BD': ...}, both trees parsed inside
```

## License

MIT
