import numpy as np
import os
import sys

from util.ply import read_ply, write_ply
import argparse
import multiprocessing

color_map = {0: [153, 151, 148], 1: [242, 207, 10], 2: [10, 242, 21], 3:[242, 10, 21], 4:[10, 41, 242], 5: [242, 150, 205]}

parser = argparse.ArgumentParser(description = "Merge blocks to form the large scale original point cloud with label and corresponding label color")
parser.add_argument('--data_path', '-d', dest = 'data_path', type = str, default='')
parser.add_argument('--predictions', '-p', dest = 'predictions', type = str, default='')
parser.add_argument('--save_path', '-s', dest = 'save_path', type = str, default='')
parser.add_argument('--original_pointcloud', '-o', default='')
args = parser.parse_args()


def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """        

    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count()-1)]

    pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)

def unpacking_apply_along_axis(all_args):
    """
    Like numpy.apply_along_axis(), but with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

def ply2array(ply_path):
    cloud = read_ply(ply_path)
    cloud = np.vstack((cloud['x'], cloud['y'], cloud['z'],cloud['red'], cloud['green'], cloud['blue'],cloud['class'])).T
    return cloud

def find_index(orig_hash, query_hash):
    idx = np.where(np.isin(orig_hash, query_hash))[0]
    return idx

def my_hash(data):
    return hash(str(data))

def row_hash(data):
    return parallel_apply_along_axis(my_hash, 1, data)

def main():
    result_path = args.predictions
    pc_list = np.load(os.path.join(result_path, 'filename.pickle'), allow_pickle=True)['label']
    preds = np.load(os.path.join(result_path, 'pred.pickle'), allow_pickle=True)['pred']
    pc_path = args.data_path


    maxkey = 0
    for key, data in color_map.items():
        if key > maxkey:
            maxkey = key
    remap_lut = np.zeros((maxkey + 100, 3), dtype=np.int32)
    for key, data in color_map.items():
        try:
            remap_lut[key] = data
        except IndexError:
            print("Wrong key ", key)
            
    orig_pc = ply2array(args.original_pointcloud)
    orig_hash = row_hash(orig_pc[:, :3])
    save_pred = np.zeros(orig_pc.shape[0])

    for i in range(len(pc_list)):
        pc = ply2array(os.path.join(pc_path, pc_list[i]+'.ply'))
        pc_hash = row_hash(pc[:, :3])
        idx = find_index(orig_hash, pc_hash)
        assert len(idx)==len(pc_hash)

        pred = preds[i].astype(np.int32)
        # pred_color = remap_lut[pred]
        save_pred[idx] = pred
    field_names = ['x', 'y', 'z', 'red', 'green', 'blue', 'pred']
    save_pred = save_pred.reshape((-1, 1))
    write_ply(args.save_path + pc_list[i][:-2] + '_all.ply', [orig_pc[:, :3], orig_pc[:, 3:6].astype(np.uint8), save_pred.astype(np.int32)], field_names)

if __name__ == "__main__":
    main()