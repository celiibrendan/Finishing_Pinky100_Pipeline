import datajoint as dj
import imageio
import matplotlib.pyplot as plt
import numpy as np

from . import ta3, nda
from .connectomics import Connected2Soma, MeshBoundingBox
from sklearn.metrics import pairwise_distances_argmin_min
from matplotlib import pyplot as plt


schema = dj.schema('microns_funconnect')


@schema
class DistanceThreshold(dj.Lookup):
    definition = """
    d_max     : Decimal(6,2)  # maximal distance to store in um
    ---
    """

    contents = [(1,)]


@schema
class MeshSurfaceArea(dj.Computed):
    definition = """
    # computes total surface area of a mesh with soma
    -> ta3.Mesh
    ---
    surface_area: float         # surface area of a mesh fragment in um^2
    """


    def make(self, key):
        vertices, triangles = (ta3.Mesh.Fragment & key).fetch('vertices', 'triangles')
        vertices = vertices / 1000  # change units from nm to um
        frag_area_list = []
        # computing area for each fragment of mesh
        for vert, triang in zip(vertices, triangles):
            tri_area_list = []
            for tri_xyz in vert[triang]:
                # computing area of triangle
                x = np.cross(tri_xyz[0] - tri_xyz[1], tri_xyz[0] - tri_xyz[2])
                area_triang = np.linalg.norm(x) / 2
                tri_area_list.append(area_triang)
            frag_area = np.sum(tri_area_list)
            frag_area_list.append(frag_area)
        mesh_area = np.sum(frag_area_list)
        key['surface_area'] = mesh_area
        self.insert1(key)
        print('Computed surface area for segment_id {segment_id}'.format(**key))


@schema
class FragementBoundingBox(dj.Computed):
    definition = """
    -> ta3.Mesh.Fragment
    ---
    x_min       : float
    x_max       : float
    y_min       : float
    y_max       : float
    z_min       : float
    z_max       : float
    """

    key_source = ta3.Mesh.Fragment & Connected2Soma

    def make(self, key):
        v = (ta3.Mesh.Fragment() & key).fetch1('vertices')
        key['x_min'], key['y_min'], key['z_min'] = np.min(v, axis=0)
        key['x_max'], key['y_max'], key['z_max'] = np.max(v, axis=0)
        self.insert1(key)


@schema
class MeshSnapshots(dj.Computed):
    definition = """
    -> ta3.Mesh
    ---
    figure: longblob   # 3d rendering
    """

    # plot figure
    @staticmethod
    def plot_figure(key, ax=None, norm_axis=False):

        key = (ta3.Mesh & key).fetch1('KEY')

        # unpacks vertices and triangles for a single mesh
        vertices, triangles = (ta3.Mesh.Fragment & key).fetch('vertices', 'triangles')

        # if user does not pass axis object then create one
        if ax is None:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
        else:
            fig = None

        # swap y and z axis for plotting in matplotlib
        vertices = [p[:, [0, 2, 1]] for p in vertices]
        triangles = [p[:, [0, 2, 1]] for p in triangles]

        # get max and min bounds from fc.MeshBoundingBox
        if norm_axis:
            attrs = {'{}_{}'.format(d, o): '{}(bound_{}_{}) / 1000'.format(o, d, o) for d in 'xyz' for o in
                     ['min', 'max']}
            x_min, y_min, z_min, x_max, y_max, z_max = dj.U('segmentation').aggr(MeshBoundingBox, **attrs).fetch1(
                *['{}_{}'.format(d, o) for o in ['min', 'max'] for d in 'xzy'])

        # plot Mesh in matplotlib
        for v, t in zip(vertices, triangles):
            ax.plot_trisurf(*v.T / 1000, triangles=t, color='red')

        # set axis according to max and mins from fc.MeshBoundingBox and center
        if norm_axis:
            ax.set_aspect('equal')

            max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0

            mid_x = (x_max + x_min) * 0.5
            mid_y = (y_max + y_min) * 0.5
            mid_z = (z_max + z_min) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_title('Segment_ID: {segment_id}, units: $\mu$m'.format(**key))

        if fig is not None:
            fig.set_dpi(300)
            fig.tight_layout()
        else:
            fig = ax.figure

        return fig, ax

    # calls plot_method() and converts image to numpy array and stores in database
    def make(self, key):
        fig, ax = self.plot_figure(key, norm_axis=True)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        key['figure'] = data

        self.insert1(key)

        fig.clf()

    def export_plot(self, path='{segmentation}_{segment_id}.png', restriction=None):
        if restriction is None:
            restriction = {}

        for key in (self & restriction).fetch('KEY'):
            data = (self & key).fetch1('figure')
            imageio.imsave(path.format(**key), data)


@schema
class SynapseVsContact(dj.Computed):
    definition = """
    -> ta3.Segment
    ---
    n_synapse       : bigint # number of synapsing axons onto this segment
    n_contact       : bigint # number of close axons for this segment
    """

    key_source = ta3.Segment & (nda.Mask & dict(mask_method=1))

    def make(self, key):

        is_axon = ta3.Neurite & dict(neurite_type='axon')

        n_contacts = len(
            ta3.Contact & [dj.AndList(['seg1={segment_id}'.format(**key), is_axon.proj(seg2='segment_id')]),
                           dj.AndList(['seg2={segment_id}'.format(**key), is_axon.proj(seg1='segment_id')])])

        n_synapses = len(
            ta3.Synapse & [dj.AndList(['postsyn={segment_id}'.format(**key), is_axon.proj(presyn='segment_id')]),
                           dj.AndList(['presyn={segment_id}'.format(**key), is_axon.proj(postsyn='segment_id')])]
        )

        key['n_contact'] = n_contacts
        key['n_synapse'] = n_synapses
        self.insert1(key)

@schema
class RFCorrelation(dj.Computed):
    definition = """
    # correlation of receptive fields
    -> ta3.Segmentation
    -> nda2.Stimulus
    ---
    segment_ids          : longblob   # vector of segment ids included in analysis
    ncomps              : int        # number of pinc components included
    rf_corr_matrix       : longblob   # correlation matrix
    """

    class Pair(dj.Part):
        definition = """
        -> master
        -> nda2.RF
        (other_seg) -> nda2.RF(segment_id)
        ---
        rf_corr_coef   :  float   # correlation coefficient between two cells
        """

    def make(self, key):
        # get RFs
        segment_ids, rfs = (nda2.RF & key).fetch('segment_id', 'rf')

        # select ROI (a mouse-specific HACK!)
        rf = np.stack(rfs, axis=-1)[0:3, 30:75, 60:120]

        # truncated singular value decomposition
        rf -= rf.mean(axis=(0,1,2), keepdims=True)
        u, d, v = np.linalg.svd(
            rf.reshape(rf.shape[0]*rf.shape[1]*rf.shape[2], rf.shape[3]),
            full_matrices=False)
        d[d.size//2:] = 0  # truncate to half
        ncomps = (d!=0).sum()

        # resynthesize RFs
        q = u@np.diag(d)@v

        # compute correlations
        corr = np.float32(np.corrcoef(q.T))
        self.insert1(dict(key,
                          ncomps=ncomps,
                          segment_ids=segment_ids,
                          rf_corr_matrix=corr))

        for i, segment_id in enumerate(segment_ids):
            self.Pair.insert(
                dict(key,
                     segment_id=segment_id,
                     other_seg=other_seg,
                     rf_corr_coef=corr[i,j])
                for j, other_seg in enumerate(segment_ids))
