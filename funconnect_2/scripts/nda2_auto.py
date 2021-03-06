"""This module was auto-generated by datajoint from an existing schema"""


import datajoint as dj

schema = dj.schema('microns_nda2')


vmodule0 = dj.create_virtual_module('vmodule0', 'microns_ta3')
vmodule1 = dj.create_virtual_module('vmodule1', 'pipeline_preprocess')
vmodule2 = dj.create_virtual_module('vmodule2', 'common_psy')


@schema
class MaskMethod(dj.Lookup):
    definition = """
    # Method for definiing two-photon masks
    mask_method          : int                          #
    ---
    method_description   : varchar(255)                 #
    methods_settings     : varchar(255)                 # in JSON format
    """


@schema
class Scan(dj.Manual):
    definition = """
    # Two-photon functional scans
    scan_idx             : smallint                     # scan ID
    ---
    depth                : int                          # Scan depth from the surface (microns)
    laser_power          : int                          # Laser power (mW)
    wavelength           : int                          # Laser wavelength (nm)
    filename             : varchar(255)                 # Scan base filename uploaded to S3
    """


@schema
class Treadmill(dj.Manual):
    definition = """
    # Treadmill activity synced to imaging
    -> Scan
    ---
    treadmill_speed      : longblob                     # vector of treadmill velocities synchronized with slice 1 frame times (cm/s)
    """


@schema
class Stimulus(dj.Manual):
    definition = """
    # Visual stimulus synced to imaging
    -> Scan
    ---
    movie                : longblob                     # stimulus images synchronized with slice 1 frame times (HxWxtimes matrix 90x160x27300)
    conditions           : longblob                     # vector indicating direction of moving noise, or NaN for stationary synchronized with slice 1 frame times (degrees)
    """


@schema
class Slice(dj.Manual):
    definition = """
    # Z-axis slice in scan
    -> Scan
    slice                : smallint                     # slice number
    ---
    z_offset             : float                        # Z-offset from scan depth (microns)
    """


@schema
class SliceShift(dj.Manual):
    definition = """
    # Pixel coordinate of Paninski Mask FOV left corner in full functional slice
    -> Slice
    ---
    xshift               : float                        # (pixels)
    yshift               : float                        # (pixels)
    """


@schema
class Mask(dj.Manual):
    definition = """
    # Cell mask in two-photon scans
    -> Slice
    -> MaskMethod
    -> vmodule0.Segment
    ---
    mask_pixels          : longblob                     # vector of pixel indices in mask
    mask_weights=null    : longblob                     # vector of pixel weights
    """


    class EASE(dj.Part):
        definition = """
        # Output of the EASE method
        -> Mask
        ---
        em_mask_pixels       : longblob                     # indices of pixels of projection of EM segment on functional slice
        em_mask_weights      : longblob                     # corresponding weights to em_mask_pixels
        confidence           : float                        # confidence score from 1 to 5.5
        """


@schema
class Trace(dj.Manual):
    definition = """
    # Raw trace extracted from cell masks
    -> Mask
    ---
    trace                : longblob                     # raw calcium trace (int16 vector)
    trace_denoised=null  : longblob                     #
    """


@schema
class Spike(dj.Manual):
    definition = """
    # inferred spike rates from traces
    -> Trace
    ---
    rate                 : longblob                     # inferred spike rate (float vector)
    """


@schema
class VonMises(dj.Computed):
    definition = """
    # directional tuning
    -> Spike
    ---
    von_r2               : float                        # r-squared explaned by vonMises fit
    von_pref             : float                        # preferred directions
    von_base             : float                        # von mises base value
    von_amp1             : float                        # amplitude of first peak
    von_amp2             : float                        # amplitude of second peak
    von_sharp            : float                        # sharpnesses
    von_pvalue           : float                        # p-value by shuffling (nShuffles = 1e4)
    """


@schema
class RF(dj.Computed):
    definition = """
    # receptive fields from Monet stimulus
    -> Stimulus
    -> Spike
    ---
    rf                   : longblob                     # receptive field map
    """


@schema
class RFScore(dj.Computed):
    definition = """
    # quality score of receptive fields
    -> RF
    ---
    score                : float                        # quality score
    """


@schema
class ScanInfo(dj.Manual):
    definition = """
    # Scanning parameters
    -> Scan
    ---
    nframes              : int                          # frames recorded
    px_width             : smallint                     # pixels per line
    px_height            : smallint                     # lines per frame
    um_width             : float                        # field of view width (microns)
    um_height            : float                        # field of view height (microns)
    bidirectional        : tinyint                      # 1=bidirectional scanning
    fps                  : float                        # frames per second (Hz)
    zoom                 : decimal(4,1)                 # zoom factor (Scanimage-specific)
    nchannels            : tinyint                      # number of recorded channels
    nslices              : tinyint                      # number of slices
    fill_fraction        : float                        # raster scan fill fraction (Scanimage-specific)
    raster_phase         : float                        # shift of odd vs even raster lines
    """


@schema
class Pupil(dj.Manual):
    definition = """
    # pupil position and size
    -> Scan
    ---
    pupil_r              : longblob                     # vector of pupil radii synchronized with slice 1 frame times (pixels)
    pupil_x              : longblob                     # vector of pupil x positions synchronized with slice 1 frame times (pixels)
    pupil_y              : longblob                     # vector of pupil y positions synchronized with slice 1 frame times (pixels)
    """


@schema
class DriftTrialSet(dj.Computed):
    definition = """
    # all drift trials for this scan
    -> vmodule1.Sync
    """


@schema
class DriftTrial(dj.Computed):
    definition = """
    # noise drift trials
    -> DriftTrialSet
    drift_trial          : smallint                     # trial index
    ---
    -> vmodule2.Trial
    direction            : float                        # (degrees) direction of drift, clock-like: 0=upward
    onset                : double                       # (s) onset time in Sync times
    offset               : double                       # (s) offset time in Sync times
    """


@schema
class DriftResponse(dj.Computed):
    definition = """
    # calcium responses to drift trials
    -> DriftTrial
    -> Spike
    ---
    response             : float                        # averaged response
    """
