import os
import csv
import pickle

def make_dir(path):
    """
    Makes a folder in provided path if it doesn't already exist
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.abspath(path)

def files_in_dir(path):
    """
    Returns a lists of all entries in provided path
    """
    return os.listdir(path)

def tifs_in_dir(path):
    """
    Returns lists containing tif paths and filename in provided path
    """
    abspath = os.path.abspath(path)
    files = files_in_dir(abspath)
    tif_paths = []
    tif_filenames = []
    for f in files:
        if f.endswith('.tif'):
            tif_paths.append(os.path.join(abspath, f))
            tif_filenames.append(f)
    return tif_paths, tif_filenames

def dict_to_csv(csvpath, input_dict):
    """
    Creates a CSV file and fills it with the data within provided dictionary
    """
    with open(csvpath, 'w', newline='') as csvfile:
        w = csv.DictWriter(csvfile, input_dict.keys())
        w.writeheader()
        w.writerow(input_dict)

def csv_to_dict(csvpath):
    """
    Opens a 2-row CSV file and returns the data as key-value pairs in a dict
    """
    with open(csvpath, newline='') as csvfile:
        r = csv.DictReader(csvfile)
        d = next(r)
    return d

def load_metadata(path):
    """
    Loads a metadata.pkl file within provided path
    """
    return pickle_load(os.path.join(path, 'metadata.pkl'))

def pickle_save(path, data):
    """
    Pickles the dictionary provided in data and saves it to provided path
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def pickle_load(path):
    """
    Un-pickles a file provided in path
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def test():
    print(len(files_in_dir('../data/spim_crop/input')))
    print(tifs_in_dir('../data/spim_crop/input'))
    metadata = load_metadata('../data/spim_crop/input')
    print(metadata)


if __name__ == '__main__':
    test()
