import os

import numpy as np 

from skimage import morphology, measure


def get_labels_from_mask(mask, do_processing=False, filter_size = None, min_obj_size=None):
    
    if do_processing:
        morp_size = filter_size if filter_size is not None else 4
        obj_size = min_obj_size if min_obj_size is not None else 200

        mask_open = morphology.binary_opening(mask, morphology.disk(morp_size))
        mask_open_close = morphology.binary_closing(mask_open, morphology.disk(morp_size))

        mask_open_close_big = morphology.remove_small_objects(mask_open_close, obj_size)
        mask_open_close_big_hole = morphology.remove_small_holes(mask_open_close_big)

        # labelize individual cells
        mask_label = measure.label(mask_open_close_big_hole)

        return mask_label
    else:
        return measure.label(mask)



def merge_labels(mask_label_1, mask_label_2):
    max_label = mask_label_1.max()

    # Offset the second image's labels
    mask_label_2_offset = np.where(mask_label_2 > 0, mask_label_2 + max_label, 0)
    # Combine the images
    merged = np.where(mask_label_1 > 0, mask_label_1, mask_label_2_offset)

    return merged


