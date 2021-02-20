# Detection of Z-disk/microtubule intersections (ZMIs)

## Description
We found that RNAs in muscle fibers are preferentially associated with microtubules and Z-disks. In addition, we believed by observation that RNAs associated with these cytoskeletal structures appeared to be enriched at or near Z-disk/microtubule intersections (ZMIs). To study this quantitatively, we developed a novel approach to detect perpendicular intersections between two feature masks.

## Instructions

## Example
First, we use the [Allen Cell and Structure Segmenter](https://www.biorxiv.org/content/10.1101/491035v2) to segment Z-disks and microtubules, and we flatten the 3D arrays into 2D using maximum intensity projection:

<img src="img/masks.png" alt="masks" width="500">
![masks](img/masks.png)

In this method, the two masks are first expanded using skeletonized