import datajoint as dj
import os
import h5py
import tqdm
import glob
import numpy as np

schema = dj.schema('microns_ta3')
schema.spawn_missing_classes()


def ingest_pyc_skeletons():

    for f in tqdm.tqdm(glob.glob('data/skeletons_pyc_180810/*')):
	key = dict(
	    segmentation=1,
	    segment_id=int(os.path.splitext(os.path.split(f)[-1])[0].split('_')[0]))
	h = h5py.File(f, 'r')
	for k in ['root', 'edges', 'nodes', 'radii']:
	    key[k] = h.get(k)[:]
	if not key['root'].size:
	    key.pop('root')
	else:
	    assert key['root'].size == 3, 'root size must be 3'
	    key['root'] = key['root'].flatten()
	Skeleton.insert1(key)


def ingest_neurite_skeletons():

    for neurite_type in ('dendrite', 'axon'):
	for f in tqdm.tqdm(glob.glob('data/skeletons_{neurite_type}s_180810/*'.format(neurite_type=neurite_type))):
	    key = dict(
		segmentation=1,
		segment_id=int(os.path.splitext(os.path.split(f)[-1])[0].split('_')[0]),
		neurite_type=neurite_type)
	    h = h5py.File(f, 'r')
	    for k in ['root', 'edges', 'nodes', 'radii']:
		key[k] = h.get(k)[:]
	    key['nroots'] = int(key['root'].size//3)
	    if not key['root'].size:
		key.pop('root')
	    else:
		assert key['root'].ndim==2 and key['root'].shape[1] == 3, 'invalid root size'
	    NeuriteSkeleton.insert1(key)


def ingest_contacts():
    data = np.genfromtxt('data/ca_0.5.complete', dtype=float, delimiter=',', names=True)
    batch = 1000
    for i in tqdm.tqdm(range(0, data.size, batch)):
        ContactIngest.insert(data[i:i+batch])


def ingest_synapse():
    data = np.genfromtxt('data/pinky40_run2_remapped.df', dtype=float, delimiter=',', names=True)
    data.dtype.names = [s.lower() for s in dat.dtype.names]
    batch = 5000
    for i in tqdm.tqdm(range(0, data.size, batch)):
        PSDIngest.insert(data[i:i+batch])
