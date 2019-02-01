from itertools import chain

import datajoint as dj
import numpy as np
import pandas as pd
from igraph import Graph
from jgraph import *
from scipy import signal
from scipy import stats
from scipy.sparse import csgraph, csc_matrix
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from . import ta3, nda

schema = dj.schema('microns_funconnect')


class SkeletonLengthMixin:
    _base_name = None
    _base = None

    @property
    def definition(self):
        return """
        ->{base_name}
        --- 
        length          : float # length (graph diameter) in um
        """.format(base_name=self._base_name)

    def make(self, key):
        nodes, edges = (self._base & key).fetch1('nodes', 'edges')
        d = np.sqrt(np.sum((nodes[edges[:, 0]] - nodes[edges[:, 1]]) ** 2, axis=1)) / 1e3  # length in um
        G = Graph(list(map(tuple, edges)))
        key['length'] = G.diameter(directed=False, weights=d)
        self.insert1(key)


@schema
class PostSynapticPairInput(dj.Computed):
    definition = """
    -> ta3.Segment
    segment_b           : bigint   # segment id unique within each Segmentation
    ---
    n_syn_a     : int # number of synapses for cell A
    n_syn_b     : int # number of synapses for cell B
    n_syn_union : int # number of unique synapses for both
    n_syn_shared: int # number of shared synapses
    n_seg_a     : int # number of segments for cell A
    n_seg_b     : int # number of segments for cell B
    n_seg_union : int # number of unique segments for both
    n_seg_shared: int # number of shared segments
    """

    @property
    def key_source(self):
        return ta3.Segmentation()

    def make(self, key):
        soma = ta3.Segment & nda.Trace & key
        axons = ta3.Neurite & 'neurite_type="axon"'
        synapse = ta3.Synapse & axons.proj(presyn='segment_id') & key
        info = soma * soma.proj(segment_b='segment_id')

        A = (synapse & soma.proj(postsyn='segment_id')).proj(
            'presyn', syn1='synapse_id', segment_id='postsyn')
        B = (synapse * dj.U('presyn') & soma.proj(postsyn='segment_id')).proj(
            'presyn', syn2='synapse_id', segment_b='postsyn')
        shared = dj.U('segment_id', 'segment_b').aggr(A * B & 'segment_id > segment_b',
                                                      n_syn_shared='count(*)',
                                                      n_seg_shared='count(DISTINCT presyn)')
        a = dj.U('segment_id').aggr(A, n_syn_a='count(*)', n_seg_a='count(DISTINCT presyn)')
        b = dj.U('segment_b').aggr(B, n_syn_b='count(*)', n_seg_b='count(DISTINCT presyn)')
        stats = (info * a * b * shared).proj('n_syn_a', 'n_syn_b', 'n_syn_shared',
                                             'n_seg_a', 'n_seg_b', 'n_seg_shared',
                                             n_syn_union='n_syn_a + n_syn_b - n_syn_shared',
                                             n_seg_union='n_seg_a + n_seg_b - n_seg_shared',
                                             )
        self.insert(stats, ignore_extra_fields=True)


@schema
class SharedInputRestriction(dj.Lookup):
    definition = """
    shared_input_type   : char(16)
    ---
    """

    @property
    def contents(self):
        yield from zip(['perisomatic', 'all', 'non-perisomatic'])


@schema
class SharedInput(dj.Computed):
    definition = """    
    -> ta3.Segment
    -> SharedInputRestriction
    segment_b           : bigint   # segment id unique within each Segmentation
    ---
    n_seg_shared        : int # number of shared segments
    """

    @property
    def key_source(self):
        return ta3.Segmentation() * SharedInputRestriction()

    def make(self, key):
        soma = ta3.Segment & nda.Trace & key
        if key['shared_input_type'] == 'all':
            axons = ta3.Neurite & 'neurite_type="axon"'
        elif key['shared_input_type'] == 'perisomatic':
            axons = ta3.Neurite & 'neurite_type="axon"' & PeriSomatic
        elif key['shared_input_type'] == 'non-perisomatic':
            axons = (ta3.Neurite & 'neurite_type="axon"') - PeriSomatic
        else:
            raise ValueError('Do not know shared_input_type={shared_input_type}'.format(**key))

        synapse = ta3.Synapse & axons.proj(presyn='segment_id') & key
        info = soma * soma.proj(segment_b='segment_id') & 'segment_id > segment_b'
        A = (synapse & soma.proj(postsyn='segment_id')).proj(
            'presyn', syn1='synapse_id', segment_id='postsyn')
        B = (synapse * dj.U('presyn') & soma.proj(postsyn='segment_id')).proj(
            'presyn', syn2='synapse_id', segment_b='postsyn')
        shared = dj.U('segment_id', 'segment_b').aggr(A * B & 'segment_id > segment_b',
                                                      n_seg_shared='count(DISTINCT presyn)')
        self.insert([dict(key, **k) for k in tqdm(shared.fetch(as_dict=True))])

        self.insert([dict(key, **k, n_seg_shared=0) for k in tqdm((info - shared).fetch(as_dict=True))],
                    ignore_extra_fields=True)


@schema
class PeriSomatic(dj.Computed):
    definition = """    
    -> ta3.Segment
    ---
    """

    @property
    def key_source(self):
        return ta3.Segmentation()

    def make(self, key):
        np.random.seed(42)
        axons = ta3.Neurite() & 'neurite_type="axon"' & key
        rel = ta3.SomaSynapseDistance() * ta3.Synapse() * axons.proj(presyn='segment_id') & key
        df = pd.DataFrame(rel.fetch())
        X = np.log(df.distance)[:, None]
        gm = GaussianMixture(n_components=2, means_init=np.array([[2, 4]]).T)
        gm.fit(X)
        p = gm.predict_proba(X)
        df['cluster'] = np.log(p[:, 0]) - np.log(p[:, 1]) > 2
        assign = df.groupby('presyn')['cluster'].mean()
        ps = np.array(assign[(assign > .5)].index)
        self.insert([dict(key, segment_id=i) for i in ps])


@schema
class PostSynapticPairAxonalInput(dj.Computed):
    definition = """
    -> ta3.Segment
    segment_b           : bigint   # segment id unique within each Segmentation
    ---
    n_syn_a     : int # number of synapses for cell A
    n_syn_b     : int # number of synapses for cell B
    n_syn_union : int # number of unique synapses for both
    n_syn_shared: int # number of shared synapses
    n_seg_a     : int # number of segments for cell A
    n_seg_b     : int # number of segments for cell B
    n_seg_union : int # number of unique segments for both
    n_seg_shared: int # number of shared segments
    """

    @property
    def key_source(self):
        return ta3.Segmentation()

    def make(self, key):
        soma = ta3.Segment & nda.Trace & key
        axons = (ta3.Neurite & 'neurite_type = "axon"').proj(presyn='segment_id')
        synapse = ta3.Synapse & key & axons
        info = soma * soma.proj(segment_b='segment_id')

        A = (synapse & soma.proj(postsyn='segment_id')).proj(
            'presyn', syn1='synapse_id', segment_id='postsyn')
        B = (synapse * dj.U('presyn') & soma.proj(postsyn='segment_id')).proj(
            'presyn', syn2='synapse_id', segment_b='postsyn')
        shared = dj.U('segment_id', 'segment_b').aggr(A * B & 'segment_id > segment_b',
                                                      n_syn_shared='count(*)',
                                                      n_seg_shared='count(DISTINCT presyn)')
        a = dj.U('segment_id').aggr(A, n_syn_a='count(*)', n_seg_a='count(DISTINCT presyn)')
        b = dj.U('segment_b').aggr(B, n_syn_b='count(*)', n_seg_b='count(DISTINCT presyn)')
        stats = (info * a * b * shared).proj('n_syn_a', 'n_syn_b', 'n_syn_shared',
                                             'n_seg_a', 'n_seg_b', 'n_seg_shared',
                                             n_syn_union='n_syn_a + n_syn_b - n_syn_shared',
                                             n_seg_union='n_seg_a + n_seg_b - n_seg_shared',
                                             )
        self.insert(stats, ignore_extra_fields=True)


@schema
class Connected2Soma(dj.Computed):
    definition = """
    -> ta3.Segment
    ---
    """

    @property
    def key_source(self):
        return ta3.Segmentation()

    def make(self, key):
        somas = ta3.Segment & nda.Trace
        presyn, postsyn = (ta3.Synapse() & key).fetch('presyn', 'postsyn')
        soma_id = (somas & key).fetch('segment_id')

        N = np.max([postsyn.max(), presyn.max()]) + 1
        G = csc_matrix((np.ones(len(presyn)), (presyn, postsyn)), shape=(N, N))

        n_components, labels = csgraph.connected_components(G, directed=True, connection='weak')
        elems = np.isin(labels, labels[soma_id])
        segments = ta3.Segment() & 'segment_id in ({})'.format(','.join(map(str, np.where(elems)[0])))
        self.insert(segments, ignore_extra_fields=True)


@schema
class ShortestPath(dj.Computed):
    definition = """
    (segmentation, source_id) -> ta3.Segment(segmentation, segment_id)
    (target_id) -> ta3.Segment
    ---
    n_paths : int       # number of paths
    d       : smallint  # degree of separation (d=synapses+1)
    """

    class Paths(dj.Part):
        definition = """
        -> master
        path_id     : int # arbitrary number of path
        ---
        path        : longblob # one of the shortest paths
        """

    @property
    def key_source(self):
        return (ta3.Segment() & nda.Trace).proj(source_id='segment_id')

    def group_paths(self, paths, **kwargs):
        paths = [self.from_index(p) for p in paths]
        path_id = 0
        old_out = None
        for p in tqdm(paths, desc='all paths'):
            if p[-1] != old_out:
                path_id = 0
            yield dict(source_id=p[0], target_id=p[-1],
                       path_id=path_id, path=np.array(p, dtype=int), **kwargs)
            path_id += 1
            old_out = p[-1]

    def pairs(self, paths, **kwargs):
        paths = [self.from_index(p) for p in paths]
        old_path = paths[0]
        n_paths = 0
        for p in tqdm(paths, desc='pairs'):
            if p[-1] != old_path[-1]:
                yield dict(source_id=old_path[0], target_id=old_path[-1],
                           d=len(old_path), n_paths=n_paths, **kwargs)
                n_paths = 0
            n_paths += 1
            old_path = p

        yield dict(source_id=old_path[0], target_id=old_path[-1],
                   d=len(old_path), n_paths=n_paths, **kwargs)

    _graph_cache = None
    _soma_id_cache = None
    _segment_map = None
    _segment_imap = None

    def cache(self, G, soma_ids):
        self._graph_cache = G
        self._soma_id_cache = soma_ids

    def cached(self):
        return self._graph_cache is not None

    def retrieve_cache(self):
        return self._graph_cache, self._soma_id_cache

    def to_index(self, iterable):
        return np.array([self._segment_map[e] for e in iterable])

    def from_index(self, iterable):
        return np.array([self._segment_imap[e] for e in iterable])

    def make(self, key):
        somas = ta3.Segment & nda.Trace
        if not self.cached():
            seg_key = dict(segmentation=key['segmentation'])
            presyn, postsyn = (ta3.Synapse() & seg_key).fetch('presyn', 'postsyn')
            soma_ids = (somas & seg_key).fetch('segment_id')

            seg_ids = list(set(chain(presyn, postsyn, soma_ids)))
            self._segment_imap = dict(zip(range(len(seg_ids)), seg_ids))
            self._segment_map = dict(zip(seg_ids, range(len(seg_ids))))

            soma_ids = self.to_index(soma_ids)
            G = Graph(list(zip(self.to_index(presyn), self.to_index(postsyn))))
            self.cache(G, soma_ids)
        else:
            G, soma_ids = self.retrieve_cache()
        paths = G.get_all_shortest_paths(self._segment_map[key['source_id']], soma_ids)
        self.insert(self.pairs(paths, segmentation=key['segmentation']))
        self.Paths.insert(self.group_paths(paths, segmentation=key['segmentation']))


@schema
class ShortestPathNode(dj.Computed):
    definition = """
    -> ta3.Segment
    ---
    """

    key_source = ta3.Segmentation()

    def make(self, key):
        paths = (ShortestPath.Paths() & key).fetch('path')
        spaths = set(np.hstack(paths))
        self.insert([dict(key, segment_id=s) for s in spaths])


@schema
class PathComposition(dj.Computed):
    definition = """
    # path type: only excitatory, only inhibitory, mixed=either inhibitory or excitatory, unknown=at least one unknown cell type
    -> ShortestPath
    ---
    """

    class Composition(dj.Part):
        definition = """
        -> master
        type    : enum('excitatory', 'inhibitory', 'mixed', 'direct', 'unknown')
        --- 
        p       : float # percentage of paths with that type
        """

    key_source = ShortestPath() & 'd > 1'

    _inhibitory_cache = set((ta3.AllenCellType & 'cell_class="inhibitory"').fetch('segment_id'))
    _excitatory_cache = set((ta3.AllenCellType & 'cell_class="excitatory"').fetch('segment_id'))

    def make(self, key):
        self.insert1(key)
        counter = dict(zip(['inhibitory', 'excitatory', 'mixed', 'unknown', 'direct'], 5 * [0]))
        for path in (ShortestPath.Paths & key).fetch('path'):
            path = set(path[1:-1])

            if not path:
                counter['direct'] += 1
            elif path - self._inhibitory_cache - self._excitatory_cache:
                counter['unknown'] += 1
            else:
                if (path & self._inhibitory_cache) and (path & self._excitatory_cache):
                    counter['mixed'] += 1
                elif path & self._inhibitory_cache:
                    counter['inhibitory'] += 1
                elif path & self._excitatory_cache:
                    counter['excitatory'] += 1

        s = sum(counter.values())
        self.Composition.insert([dict(key, type=k, p=n / s) for k, n in counter.items()])


@schema
class NeuriteLength(SkeletonLengthMixin, dj.Computed):
    _base_name = 'ta3.NeuriteSkeleton'
    _base = ta3.NeuriteSkeleton


@schema
class MeshBoundingBox(dj.Computed):
    definition = """
    -> ta3.Mesh
    ---
    bound_x_min: float
    bound_x_max: float
    bound_y_min: float
    bound_y_max: float
    bound_z_min: float
    bound_z_max: float
    """

    class FragmentBoundingBox(dj.Part):
        definition = """
        -> master
        -> ta3.Mesh.Fragment
        ---
        fragment_bound_x_min: float
        fragment_bound_x_max: float
        fragment_bound_y_min: float
        fragment_bound_y_max: float
        fragment_bound_z_min: float
        fragment_bound_z_max: float
        """

    def make(self, key):
        fragment_keys, fragments = (ta3.Mesh.Fragment & key).fetch('KEY', 'vertices')
        f_mins = []
        f_maxs = []
        for f in fragments:
            f_mins.append(f.min(axis=0))
            f_maxs.append(f.max(axis=0))

        mesh_mins = np.stack(f_mins).min(axis=0)
        mesh_maxs = np.stack(f_maxs).max(axis=0)

        key['bound_x_min'] = mesh_mins[0]
        key['bound_y_min'] = mesh_mins[1]
        key['bound_z_min'] = mesh_mins[2]

        key['bound_x_max'] = mesh_maxs[0]
        key['bound_y_max'] = mesh_maxs[1]
        key['bound_z_max'] = mesh_maxs[2]

        self.insert1(key)

        for fkey, fmin, fmax in zip(fragment_keys, f_mins, f_maxs):
            fkey['fragment_bound_x_min'] = fmin[0]
            fkey['fragment_bound_y_min'] = fmin[1]
            fkey['fragment_bound_z_min'] = fmin[2]
            fkey['fragment_bound_x_max'] = fmax[0]
            fkey['fragment_bound_y_max'] = fmax[1]
            fkey['fragment_bound_z_max'] = fmax[2]
            self.FragmentBoundingBox.insert1(fkey)


@schema
class TargetFrequency(dj.Lookup):
    definition = """
    # values of target frequency used in Hamming filter
    target_freq          : int        # Frequency that spike trace will be downsampled to after filtering with Hamming
    ---
    """


@schema
class FunCorrelation(dj.Computed):
    definition = """
    # correlation of traces for all cells
    (scan_idx, segmentation, presyn) -> nda.Spike(scan_idx, segmentation, segment_id)
    (postsyn) -> nda.Spike(segment_id)
    ->TargetFrequency          
    ---
    correlation          : float      # Pearson's correlation coefficient
    p_val                : float      # two-tailed p_value of slope of the linear regression
    """

    @property
    def key_source(self):
        return (nda.Spike.proj(presyn='segment_id') * nda.Spike.proj(
            postsyn='segment_id') & 'presyn != postsyn') * TargetFrequency

    def make(self, key):
        pre_trace = (nda.Spike.proj('rate', presyn='segment_id') & key).fetch1('rate').ravel()
        post_trace = (nda.Spike.proj('rate', postsyn='segment_id') & key).fetch1('rate').ravel()

        fps = (nda.ScanInfo & key).fetch1('fps')
        target_freq = (TargetFrequency() & key).fetch1('target_freq')
        window = 2 * int(fps / target_freq) + 1
        ham_filter = signal.hamming(window)

        pre_trace_conv = np.convolve(pre_trace, ham_filter, mode='valid')
        post_trace_conv = np.convolve(post_trace, ham_filter, mode='valid')

        key['correlation'], key['p_val'] = stats.pearsonr(pre_trace_conv, post_trace_conv)

        self.insert1(key)


@schema
class SharedInputPairs(dj.Computed):
    definition = """
    (seg1)->ta3.Segment(segment_id)
    (seg2)->ta3.Segment(segment_id)
    ---
    axon_segs   : longblob  # vector of axon segment_ids
    n_syn_1     : longblob  # vector of synapses from axon_segs to seg1
    n_con_1     : longblob  # vector of contacts between axon_segs and seg1
    n_syn_2     : longblob  # vector of synapses from axon_segs to seg2
    n_con_2     : longblob  # vector of contacts between axon_segs and seg2
    """

    @property
    def key_source(self):
        return (ta3.Segment & nda.Trace).proj(seg1='segment_id') * (ta3.Segment & nda.Trace).proj(
            seg2='segment_id') & 'seg1<seg2'

    @staticmethod
    def ordered_segs(seg_ref, sub_segs, nums):
        ordered_segs = np.zeros_like(seg_ref)
        for sub_seg, num in zip(sub_segs, nums):
            ordered_segs[np.where(seg_ref == sub_seg)] = num
        return ordered_segs

    def make(self, key):
        seg1 = key['seg1']
        seg2 = key['seg2']

        axons = ta3.Neurite() & 'neurite_type = "axon"'
        axon_segs = axons.fetch('segment_id', order_by='segment_id')

        presyns, syn_nums = axons.proj(presyn='segment_id').aggr(ta3.Synapse & {'postsyn': seg1},
                                                                 syn_num='count(synapse_id)').fetch('presyn', 'syn_num')
        n_syn_1 = self.ordered_segs(axon_segs, presyns, syn_nums)

        presyns, con_nums = axons.proj(seg2='segment_id').aggr((ta3.Contact & {'seg1': seg1}),
                                                               con_num='count(contact_num)').fetch('seg2', 'con_num')
        n_con_1a = self.ordered_segs(axon_segs, presyns, con_nums)

        presyns, con_nums = axons.proj(seg1='segment_id').aggr((ta3.Contact & {'seg2': seg1}),
                                                               con_num='count(contact_num)').fetch('seg1', 'con_num')
        n_con_1b = self.ordered_segs(axon_segs, presyns, con_nums)

        n_con_1 = np.sum((n_con_1a, n_con_1b), axis=0)

        presyns, syn_nums = axons.proj(presyn='segment_id').aggr(ta3.Synapse & {'postsyn': seg2},
                                                                 syn_num='count(synapse_id)').fetch('presyn', 'syn_num')
        n_syn_2 = self.ordered_segs(axon_segs, presyns, syn_nums)

        presyns, con_nums = axons.proj(seg2='segment_id').aggr((ta3.Contact & {'seg1': seg2}),
                                                               con_num='count(contact_num)').fetch('seg2', 'con_num')
        n_con_2a = self.ordered_segs(axon_segs, presyns, con_nums)

        presyns, con_nums = axons.proj(seg1='segment_id').aggr((ta3.Contact & {'seg2': seg2}),
                                                               con_num='count(contact_num)').fetch('seg1', 'con_num')
        n_con_2b = self.ordered_segs(axon_segs, presyns, con_nums)

        n_con_2 = np.sum((n_con_2a, n_con_2b), axis=0)

        tup = {**key, 'axon_segs': axon_segs, 'n_syn_1': n_syn_1, 'n_con_1': n_con_1, 'n_syn_2': n_syn_2,
               'n_con_2': n_con_2}

        self.insert1(tup)


@schema
class NonSynapticAxonTraceContact(dj.Manual):
    definition = """
    (axon) -> ta3.Segment(segment_id)
    (trace) -> ta3.Segment(segment_id)
    contact_num          : smallint                     # contact number between given segments
    ---
    x_faces              : int                          # voxel faces on x
    y_faces              : int                          # voxel faces
    z_faces              : int                          # voxel faces
    contact_x            : double                       # centroid location x
    contact_y            : double                       # centroid location y
    contact_z            : double                       # centroid location z
    con_bbox_x_axon          : bigint                       # bounding box corner axon x
    con_bbox_y_axon          : bigint                       # bounding box corner axon y
    con_bbox_z_axon          : bigint                       # bounding box corner axon z
    con_bbox_x_trace          : bigint                       # bounding box corner trace x
    con_bbox_y_trace          : bigint                       # bounding box corner trace y
    con_bbox_z_trace          : bigint                       # bounding box corner trace z
    """

    def fill(self):
        A = (ta3.Contact & (ta3.Neurite & 'neurite_type = "axon"').proj(seg2='segment_id') & nda.Trace.proj(
            seg1='segment_id') & 'seg1 != seg2') - ta3.Synapse.proj(seg1='postsyn', seg2='presyn')
        B = A.proj('x_faces', 'y_faces', 'z_faces', 'contact_x', 'contact_y', 'contact_z', axon='seg2', trace='seg1',
                   con_bbox_x_axon='con_bbox_x2', con_bbox_y_axon='con_bbox_y2', con_bbox_z_axon='con_bbox_z2',
                   con_bbox_x_trace='con_bbox_x1', con_bbox_y_trace='con_bbox_y1', con_bbox_z_trace='con_bbox_z1')

        self.insert(B)

        C = (ta3.Contact & (ta3.Neurite & 'neurite_type = "axon"').proj(seg1='segment_id') & nda.Trace.proj(
            seg2='segment_id')) -  ta3.Synapse.proj(seg2='postsyn', seg1='presyn')
        D = C.proj('x_faces', 'y_faces', 'z_faces', 'contact_x', 'contact_y', 'contact_z', axon='seg1', trace='seg2',
                   con_bbox_x_axon='con_bbox_x1', con_bbox_y_axon='con_bbox_y1', con_bbox_z_axon='con_bbox_z1',
                   con_bbox_x_trace='con_bbox_x2', con_bbox_y_trace='con_bbox_y2', con_bbox_z_trace='con_bbox_z2')

        self.insert(D)


@schema
class SharedContactPair(dj.Computed):
    definition = """
    (segment_a) -> ta3.Segment(segment_id)
    (segment_b) -> ta3.Segment(segment_id) # segment id unique within each Segmentation
    ---
    n_axon_a     : int # number of axon segments for cell A
    n_axon_b     : int # number of segments for cell B
    n_axon_union : int # number of unique segments for both
    n_axon_shared: int # number of shared segments
    """

    @property
    def key_source(self):
        return ta3.Segmentation()

    def make(self, key):
        trace = ta3.Segment & nda.Trace & key  # grab all segments with traces
        contact = NonSynapticAxonTraceContact & key  # grab all axon to trace contacts
        trace_a = trace.proj(segment_a='segment_id')
        trace_b = trace.proj(segment_b='segment_id')
        info = trace_a * trace_b  # compute matrix of all pairs of traces

        A = (contact & trace.proj(trace='segment_id')).proj(segment_a='trace')
        B = (contact & trace.proj(trace='segment_id')).proj(segment_b='trace')

        shared = info.aggr(A * B & 'segment_b > segment_a', n_axon_shared='count(DISTINCT axon)')
        a = trace_a.aggr(A, n_axon_a='count(DISTINCT axon)')
        b = trace_b.aggr(B, n_axon_b='count(DISTINCT axon)')
        stats = (a * b * shared).proj('n_axon_a', 'n_axon_b', 'n_axon_shared',
                                      n_axon_union='n_axon_a + n_axon_b - n_axon_shared')
        self.insert(stats)

        # for those pairs without a shared axon, enter 0 for n_axon_shared
        nonshared_info = ((trace_a * trace_b & 'segment_b > segment_a') - self.proj()).proj(n_axon_shared='0')
        stats = (nonshared_info * a * b).proj('n_axon_a', 'n_axon_b', 'n_axon_shared',
                                              n_axon_union='n_axon_a + n_axon_b')
        self.insert(stats)
