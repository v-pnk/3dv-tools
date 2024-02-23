#!/usr/bin/env python3


"""
Divide a COLMAP model into multiple parts based on image covisibility

Modes:
- fixed_k - divide into fixed number of k clusters using normalized cut
- stop_at_min - divide each cluster into two by normalized cut until all the clusters have max. max_images - do not divide if one of the two resulting subclusters has less than min_images
- divide_and_merge - divide each cluster into two by normalized cut until all the clusters have max. max_images, then merge clusters so that none has less than min_images
- bin_agg_clust - create binary tree of images and apply agglomerative clustering
"""


import os
import copy

import argparse
from tqdm import tqdm
import numpy as np
import pycolmap
from sklearn.cluster import SpectralClustering


parser = argparse.ArgumentParser()
parser.add_argument(
    "input_colmap", 
    type=str, 
    help="Input COLMAP model directory"
)
parser.add_argument(
    "output_dir", 
    type=str, 
    help="Output directory for the divided COLMAP models"
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["fixed_k", "stop_at_min", "divide_and_merge", "bin_agg_clust"],
    default="bin_agg_clust",
    help="Partitioning mode: stop division if minimum number of images is hit or divide to small parts and merge then merge to achieve the minimum number of images",
)
parser.add_argument(
    "--min_shared_pnts",
    type=int,
    default=5,
    help="Minimum number of shared 3D points between two images to be considered covisible",
)
parser.add_argument(
    "--covis_fnc",
    type=str,
    choices=["lin", "tanh10", "tanh20", "tanh50", "tanh100"],
    default="lin",
    help="Function used on top of the covisibility matrix",
)
parser.add_argument(
    "-k",
    "--n_clusters",
    type=int,
    help="Number of submodels to divide the input model into",
)
parser.add_argument(
    "--max_images",
    type=int,
    help="Maximum number of images in single cluster (overrides -k/--n_clusters)",
)
parser.add_argument(
    "--min_images", 
    type=int, 
    help="Minimum number of images in single cluster"
)
parser.add_argument(
    "--agg_clust_divider",
    type=float,
    default=8,
    help="Covisibility score divider for the agglomerative clustering (1 --> sum, 2 --> mean, ...)",
)
parser.add_argument(
    "--overlap_perc",
    type=float,
    default=0.0,
    help="Mean percentage of overlap between submodels (the number of overlapping images will be total_imgs_num/k*overlap_perc)",
)
parser.add_argument(
    "--overlap_abs",
    type=int,
    default=0,
    help="Absolute number of overlapping images for each submodels (overrides --overlap_perc)",
)
parser.add_argument(
    "--norm_covis_by_dist",
    action="store_true",
    help="normalize the covisibility score by distance between the cameras",
)
parser.add_argument(
    "--covis_matrix_path",
    type=str,
    help="Optional path to the covisibility matrix (for both loading and saving)",
)


def main(args):
    assert os.path.isdir(
        args.input_colmap
    ), 'Input COLMAP model directory "{}" does not exist'.format(args.input_colmap)
    assert os.path.isdir(
        args.output_dir
    ), 'Output directory "{}" does not exist'.format(args.output_dir)

    # - read the input COLMAP model and create list of image IDs
    print("- reading the input COLMAP model")
    model_in = pycolmap.Reconstruction(args.input_colmap)
    img_ids = list(model_in.images.keys())

    # - build the covisibility matrix
    #   - each element is the number of 3D points visible from both images
    print("- building the covisibility matrix")

    if args.covis_matrix_path is not None and os.path.isfile(args.covis_matrix_path):
        print("- loading the covisibility matrix")
        covis_matrix = np.load(args.covis_matrix_path)
    else:
        covis_matrix = np.zeros((len(img_ids), len(img_ids)), dtype=np.uint32)

        for point3D in tqdm(model_in.points3D.values()):
            track_img_idxs = [
                img_ids.index(trck_elem.image_id)
                for trck_elem in point3D.track.elements
            ]
            track_img_pairs = [
                (a, b)
                for idx, a in enumerate(track_img_idxs)
                for b in track_img_idxs[idx + 1 :]
            ]
            for a_idx, b_idx in track_img_pairs:
                covis_matrix[a_idx, b_idx] += 1
                covis_matrix[b_idx, a_idx] += 1

        if args.covis_matrix_path is not None and not os.path.isfile(
            args.covis_matrix_path
        ):
            print("- saving the covisibility matrix")
            np.save(args.covis_matrix_path, covis_matrix)

    if args.covis_fnc.startswith("tanh"):
        # - limits the covisibility matrix values to lim_pnts_num and applies tanh
        lim_pnts_num = int(args.covis_fnc[4:])
        covis_matrix = lim_pnts_num * np.tanh(covis_matrix / lim_pnts_num)

    if args.norm_covis_by_dist:
        c_dict = {}
        c_all = np.empty((3, 0))
        for img_id, img in model_in.images.items():
            c = img.projection_center()
            c_dict[img_id] = c
            c_all = np.append(c_all, np.reshape(c, (3, 1)), axis=1)
        c_dist_matrix = dist_arr(c_all)
        c_dist_max = np.amax(c_dist_matrix)
        c_dist_matrix = c_dist_matrix / c_dist_max
        covis_matrix = covis_matrix * (1 - c_dist_matrix)

    # - delete connections with less than given number of shared 3D points
    print("- filtering the covisibility matrix based on number of shared 3D points")
    covis_matrix[covis_matrix < args.min_shared_pnts] = 0

    # - partition images into submodels
    print("- partitioning the images")
    if args.mode == "fixed_k":
        print("  - using fixed number of clusters: {}".format(args.n_clusters))
        assert args.n_clusters is not None
        sc = SpectralClustering(
            n_clusters=args.n_clusters, affinity="precomputed", n_jobs=-1
        )
        labels = sc.fit_predict(covis_matrix)

        submodel_img_idxs = [[] for _ in range(args.n_clusters)]
        for img_idx, label in enumerate(labels):
            submodel_img_idxs[label].append(img_idx)
    elif args.mode in ["stop_at_min", "divide_and_merge"]:
        print(
            "  - dividing into submodels with max. {} images each".format(
                args.max_images
            )
        )
        assert args.max_images is not None
        max_images_curr = len(img_ids)
        submodel_img_idxs = [list(range(len(img_ids)))]
        submodel_img_idxs_skip = []

        while max_images_curr > args.max_images:
            max_images_curr = 0
            submodel_img_idxs_new = []
            for submodel_idx, curr_img_idxs in enumerate(submodel_img_idxs):
                if len(curr_img_idxs) > args.max_images:
                    covis_matrix_curr = covis_matrix[
                        np.ix_(curr_img_idxs, curr_img_idxs)
                    ]
                    sc = SpectralClustering(
                        n_clusters=2, affinity="precomputed", n_jobs=-1
                    )
                    labels_curr = sc.fit_predict(covis_matrix_curr)
                    submodel_1 = [
                        curr_img_idxs[i]
                        for i, label in enumerate(labels_curr)
                        if label == 0
                    ]
                    submodel_2 = [
                        curr_img_idxs[i]
                        for i, label in enumerate(labels_curr)
                        if label == 1
                    ]

                    if args.min_images is not None and args.mode == "stop_at_min":
                        if (
                            len(submodel_1) < args.min_images
                            or len(submodel_2) < args.min_images
                        ):
                            print(
                                "  - {} --> {} + {} - too small --> keeping original".format(
                                    len(curr_img_idxs), len(submodel_1), len(submodel_2)
                                )
                            )
                            submodel_img_idxs_skip.append(curr_img_idxs)
                            continue

                    submodel_img_idxs_new += [submodel_1, submodel_2]
                    print(
                        "  - {} --> {} + {}".format(
                            len(curr_img_idxs), len(submodel_1), len(submodel_2)
                        )
                    )
                    max_images_curr = max(
                        max_images_curr, max(len(submodel_1), len(submodel_2))
                    )
                else:
                    submodel_img_idxs_new.append(curr_img_idxs)
            submodel_img_idxs = submodel_img_idxs_new
        submodel_img_idxs += submodel_img_idxs_skip

        labels = np.zeros(len(img_ids), dtype=np.uint32)
        for submodel_idx, curr_img_idxs in enumerate(submodel_img_idxs):
            labels[curr_img_idxs] = submodel_idx
        print("  - model partitioned into {} submodels".format(len(submodel_img_idxs)))

        if args.mode == "divide_and_merge":
            # - the model is now divided into submodels with max.
            #   args.max_images images each --> merge the submodels to achieve
            #   the minimum number of images
            print("  - merging submodels to achieve minimum number of images")

            assert False, "NIY"
            # TODO

    elif args.mode == "bin_agg_clust":
        print("  - running agglomerative clustering on binary tree")
        # TODO
        # Create the binary tree of images based on covisibility
        # - find maximum in covisibility matrix
        # - merge the maximum pair
        # - adjust the covisibility matrix so that the pair is now represented by single node and the covisibility to outside nodes is sum/mean of covisibility of the images from the pair

        covis_matrix_agg = copy.copy(covis_matrix)
        np.fill_diagonal(covis_matrix_agg, 0)
        img_tree = list(range(len(img_ids)))

        max_images_curr = len(img_ids)
        # while len(img_tree) > 1:
        while len(img_tree) > args.n_clusters:
            a, b = np.unravel_index(np.argmax(covis_matrix_agg), covis_matrix_agg.shape)
            covis_ab = covis_matrix_agg[a, b]

            # - update the covisibility matrix
            covis_matrix_agg = np.delete(covis_matrix_agg, [a, b], axis=1)
            covis_vect_ab = (
                covis_matrix_agg[a, :] + covis_matrix_agg[b, :]
            ) / args.agg_clust_divider
            covis_matrix_agg = np.delete(covis_matrix_agg, [a, b], axis=0)
            covis_matrix_agg = np.append(
                covis_matrix_agg, np.reshape(covis_vect_ab, (1, -1)), axis=0
            )
            covis_vect_ab = np.append(covis_vect_ab, 0)
            # covis_vect_ab = np.append(covis_vect_ab, covis_ab)
            covis_matrix_agg = np.append(
                covis_matrix_agg, np.reshape(covis_vect_ab, (-1, 1)), axis=1
            )

            # - update the image tree
            a_elem = img_tree[a]
            b_elem = img_tree[b]
            img_tree = [elem for idx, elem in enumerate(img_tree) if idx not in [a, b]]
            img_tree.append([a_elem, b_elem])

            print(len(img_tree))

        # - create the submodels
        submodel_img_idxs = []
        for img_tree_elem in img_tree:
            submodel_img_idxs.append(flatten_list(img_tree_elem))
        labels = np.zeros(len(img_ids), dtype=np.uint32)
        for submodel_idx, curr_img_idxs in enumerate(submodel_img_idxs):
            labels[curr_img_idxs] = submodel_idx

    # - add overlaps
    total_overlaps = 0
    k = len(submodel_img_idxs)
    if args.overlap_perc > 0 or args.overlap_abs > 0:
        print("- adding overlaps")
        if args.overlap_abs > 0:
            overlap_imgs_num = args.overlap_abs
        else:
            overlap_imgs_num = int(len(img_ids) / k * args.overlap_perc)
        submodel_img_idxs_overlap = [[] for _ in range(k)]
        for submodel_idx, curr_img_idxs in enumerate(submodel_img_idxs):
            oosm_list = []
            for img_idx in curr_img_idxs:
                # - get the indices of the images that are covisible with the current image and are outside of its submodel
                oosm_img_idxs = np.where(
                    (covis_matrix[img_idx, :] > 0) & (labels != submodel_idx)
                )[0]
                oosm_img_covis = covis_matrix[img_idx, oosm_img_idxs]

                for oosm_img_idx, oosm_img_covis in zip(oosm_img_idxs, oosm_img_covis):
                    oosm_list.append(
                        {
                            "insm_img_idx": img_idx,
                            "oosm_img_idx": oosm_img_idx,
                            "oosm_img_covis": oosm_img_covis,
                        }
                    )

            # - sort the list of out-of-submodel images by covisibility
            oosm_list.sort(key=lambda x: x["oosm_img_covis"], reverse=True)

            # - take the top overlap_imgs_num images from the list
            oosm_list = oosm_list[:overlap_imgs_num]
            total_overlaps += len(oosm_list)

            # - create new submodel including the out-of-submodel images
            submodel_img_idxs_overlap[submodel_idx] = curr_img_idxs.copy()
            for oosm_elem in oosm_list:
                submodel_img_idxs_overlap[submodel_idx].append(
                    oosm_elem["oosm_img_idx"]
                )

        submodel_img_idxs = submodel_img_idxs_overlap

        print("  - added in total {} overlapping cameras".format(total_overlaps))

    # - get image IDs for each submodel
    submodel_img_ids = [[] for _ in range(k)]
    for submodel_idx, img_idxs in enumerate(submodel_img_idxs):
        submodel_img_ids[submodel_idx] = [img_ids[img_idx] for img_idx in img_idxs]

    # - create a submodel for each submodel image ID list
    print("- creating the submodels")
    for submodel_idx, submodel_img_id_list in enumerate(submodel_img_ids):
        print(
            "  - creating submodel {:0>3d} with {} images".format(
                submodel_idx, len(submodel_img_id_list)
            )
        )
        model_out = copy.deepcopy(model_in)
        for img in model_out.images.values():
            if img.image_id not in submodel_img_id_list:
                model_out.deregister_image(img.image_id)

        out_dir = os.path.join(args.output_dir, "submodel_{:0>3d}".format(submodel_idx))
        os.makedirs(out_dir, exist_ok=True)
        model_out.write(out_dir)


def dist_arr(pos_set):
    n = pos_set.shape[1]
    set_tile_c = np.tile(np.reshape(pos_set.T, (-1, 1, 3)), (1, n, 1))
    set_tile_r = np.tile(np.reshape(pos_set.T, (1, -1, 3)), (n, 1, 1))
    dist_arr = np.squeeze(np.sqrt(np.sum((set_tile_c - set_tile_r) ** 2, axis=2)))
    return dist_arr


def count_size(list_tree):
    if isinstance(list_tree, list):
        return sum([count_size(elem) for elem in list_tree])
    else:
        return 1


def flatten_list(list_tree):
    if isinstance(list_tree, list):
        return [elem for sublist in list_tree for elem in flatten_list(sublist)]
    else:
        return [list_tree]


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
