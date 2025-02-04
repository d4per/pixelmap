import numpy as np
from scipy.spatial import cKDTree

class CircularFeatureDescriptorMatcher:
    """
    Provides a way to match circular feature descriptors between two images,
    using a KD-tree for efficient nearest-neighbor lookups.
    """

    def __init__(self):
        pass

    def match_areas(self, img1, img2):
        """
        Builds a KD-tree from the descriptors in img1, then for each descriptor
        in img2 finds its nearest neighbor in that tree.

        Parameters:
            img1: A 'CircularFeatureGrid' (or similar) that provides
                  a list of CircularFeatureDescriptor objects.
            img2: Another 'CircularFeatureGrid'.

        Returns:
            A list of pairs (desc1, desc2), where desc1 is the nearest match
            to desc2 in feature space.
        """
        # 1. Get the list of descriptors from the first image
        descriptors1 = img1.get_infos()  # Should return a list of CircularFeatureDescriptor
        # Convert feature_vector into a NumPy array of shape (N, 6).
        data1 = np.array([d.feature_vector for d in descriptors1], dtype=np.int64)

        # 2. Build a KD-tree on the first image's feature vectors.
        kdtree = cKDTree(data1)

        # 3. For each descriptor in the second image, find the nearest neighbor in the KD-tree.
        descriptors2 = img2.get_infos()

        matched_pairs = []
        for d2 in descriptors2:
            # Convert d2's feature vector to the same type (int64 array).
            query_vec = np.array(d2.feature_vector, dtype=np.int64)

            # cKDTree.query returns (distance, index)
            dist, idx = kdtree.query(query_vec)

            # If idx is valid, retrieve the matching descriptor from descriptors1
            if idx is not None and 0 <= idx < len(descriptors1):
                d1 = descriptors1[idx]
                matched_pairs.append((d1, d2))

        return matched_pairs
