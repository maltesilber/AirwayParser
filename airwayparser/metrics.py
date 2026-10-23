import numpy as np
from .skeleton.parser import largest_component, parse_tree
    

def voxel_skeleton(tree, shape):
    # round branch points to nearest voxel coordinates
    # note: might not be connected in voxel grid
    skeleton = np.full(shape, -1, np.int32)
    for b in tree.branches.values():
        vox = tuple(np.rint(b.points).astype(int).T)
        skeleton[vox] = np.where(skeleton[vox] < 0, b.id, skeleton[vox])
    return skeleton


def dice(prediction, reference):
    p, r = np.asarray(prediction, bool), np.asarray(reference, bool)
    return float(2.0 * np.count_nonzero(p & r) / (np.count_nonzero(p) + np.count_nonzero(r)))


def precision(prediction, reference):
    p, r = np.asarray(prediction, bool), np.asarray(reference, bool)
    return float(np.count_nonzero(p & r) / np.count_nonzero(p))


def tree_length_detected(prediction, skeleton):
    return float(np.count_nonzero(prediction[skeleton >= 0]) / np.count_nonzero(skeleton >= 0))


def branches_detected(prediction, skeleton, threshold=0.8):
    ids = skeleton[skeleton >= 0]
    branch, total = np.unique(ids, return_counts=True)
    inside = np.bincount(ids[np.asarray(prediction, bool)[skeleton >= 0]], minlength=branch[-1] + 1)[branch]
    detected = int(np.count_nonzero(inside / total >= threshold))
    return detected / len(branch)


def cl_dice(prediction, reference, skeleton, prediction_skeleton):
    sens, prec = tree_length_detected(prediction, skeleton), tree_length_detected(reference, prediction_skeleton)
    return 2.0 * prec * sens / (prec + sens) if prec + sens else 0.0

def evaluate(prediction, reference, spacing=(1.0, 1.0, 1.0), threshold=0.8):
    prediction, reference = np.asarray(prediction, bool), np.asarray(reference, bool)
    largest = largest_component(prediction, np.ones((3, 3, 3), bool))
    
    skeleton = voxel_skeleton(parse_tree(reference, spacing, verbose=False), reference.shape)
    prediction_skeleton = voxel_skeleton(parse_tree(prediction, spacing, verbose=False), prediction.shape)

    return {"DSC": dice(prediction, reference), 
            "Precision": precision(prediction, reference),
            "clDice": cl_dice(prediction, reference, skeleton, prediction_skeleton),
            "TLD": tree_length_detected(largest, skeleton), 
            "BD": branches_detected(largest, skeleton, threshold)
    }