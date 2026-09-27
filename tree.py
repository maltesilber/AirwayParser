from dataclasses import dataclass, field
import numpy as np


@dataclass
class Branch:
    id: int
    parent: int | None
    generation: int
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))  # centerline in voxel coordinates, root end first
    children: list = field(default_factory=list)

    @property
    def start_point(self):
        return self.points[0]

    @property
    def end_point(self):
        return self.points[-1]

    @property
    def is_leaf(self):
        return not self.children

class BranchTree:
    def __init__(self, spacing=(1.0, 1.0, 1.0)):
        self.branches = {}
        self.spacing = tuple(float(s) for s in spacing)

    def add_branch(self, parent, generation):
        b = Branch(len(self.branches), parent, generation)
        self.branches[b.id] = b
        if parent is not None:
            self.branches[parent].children.append(b.id)
        return b

    def root(self):
        return self.branches[0]

    def leaves(self):
        return [b for b in self.branches.values() if b.is_leaf]

    def branch_points(self):
        return [b.end_point for b in self.branches.values() if not b.is_leaf]
