#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A GUI for the galaxy kinematics modeling and fitting tool DysmalPy.

Last updates:
    - 2021-06-23 initialized by Daizhong Liu.
    - 2021-06-28 first version finalized by Daizhong Liu.
    - 2021-07-06 small adjustments, e.g., data points, panel widgets
    - 2021-07-25 adding lensing
    - 2021-08-03 xshift yshift can be free
    - 2021-08-04 can save lensing source plane and image plane cubes
    - 2021-08-05 change logging.DEBUG to logging.INFO
    - 2021-08-12 substantially rewriting to use multiprocessing as scipy.linalg.inv breaks in QThread
    - 2022-08-17 adding non-circular motion, LineEditModelParamsDictForNonCirc
    - 2022-12-02 adding utils_cube_populating, zcalc_with_c, psf_fwhm_major
    - 2023-01-30 try import dysmal first so that this script can be run from anywhere.
    - 2024-08-30 updated for new dysmalpy
    - 2025-08-27 
    
Issues:
    - 2021-08-12 (solved) 3D cube fitting breaks, tracing the error to
                 astropy.modeling.fitting.LevMarLSQFitter `optimize.leastsq`,
                 scipy.optimize.minpack.leastsq `cov_x = inv(dot(transpose(R), R))`,
                 scipy.linalg.inv,
                 scipy.linalg.basic `getrf`. Have to use multiprocessing and "spawn".
    - 2021-08-12 for 3D cube fitting, plot_spaxel_compare_3D_cubes for a 100x100x80 cube took
                 nearly 1 hour!
    - 2023-03-06 input param file must ends with ".params"
    
"""

import os, sys, re, copy, json, time, datetime, ast, shutil, operator, base64, traceback
import numpy as np
from enum import Enum
from pprint import pprint
from collections import OrderedDict, namedtuple
from scipy.interpolate import griddata, interp2d

import logging
#from logutils.queue import QueueHandler, QueueListener
from logging.handlers import QueueHandler, QueueListener
#logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
#print('__name__', __name__)
logging.basicConfig()
#import multiprocessing_logging
#multiprocessing_logging.install_mp_handler()
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

#import multiprocessing
import dill, multiprocessing
multiprocessing.reduction.ForkingPickler.dumps = dill.dumps
from queue import Empty as QueueEmpty
if sys.version_info >= (3, 9):
    from multiprocessing.managers import SharedMemoryManager
elif sys.version_info >= (3, 8):
    from multiprocessing.managers import SharedMemoryManager
    from multiprocessing.managers import Server, SharedMemoryServer
    def create(self, c, typeid, *args, **kwargs):
        if hasattr(self.registry[typeid][-1], "_shared_memory_proxy"):
            kwargs['shared_memory_context'] = self.shared_memory_context
        return Server.create(self, c, typeid, *args, **kwargs)
    SharedMemoryServer.create = create
    # see https://stackoverflow.com/questions/59172691/why-do-we-get-a-nameerror-when-trying-to-use-the-sharedmemorymanager-python-3-8
else:
    from multiprocessing.managers import BaseManager as SharedMemoryManager
import threading
from threading import Thread

try:
    import dysmalpy
except:
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #print('script_dir:', script_dir)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        #print('sys.path:', sys.path)
    import dysmalpy

from dysmalpy import galaxy, models, fitting, instrument, parameters, plotting, config, data_classes, aperture_classes
#from dysmalpy.fitting_wrappers.plotting import plot_bundle_1D, plot_bundle_2D #<20240821># function no longer valid
from dysmalpy.fitting_wrappers.dysmalpy_fit_single import dysmalpy_fit_single
#<20240830>#from dysmalpy.fitting_wrappers.setup_gal_models import (setup_gal_model_base,
#<20240830>#        setup_single_object_1D, setup_single_object_2D, setup_single_object_3D,
#<20240830>#        setup_fit_dict, setup_lensing_dict)
from dysmalpy.fitting_wrappers.setup_gal_models import (setup_gal_model_base, 
         setup_single_galaxy, 
         make_empty_observation, setup_instrument_params, setup_basic_aperture_types, 
         load_galaxy)
#from dysmalpy.fitting_wrappers import data_io
from dysmalpy.fitting_wrappers import utils_io
from dysmalpy.fitting_wrappers import data_io as fwdata_io
from dysmalpy import data_io
from dysmalpy.utils import apply_smoothing_3D, rebin, gaus_fit_sp_opt_leastsq
ensure_path_trailing_slash = data_io.ensure_path_trailing_slash
#plot_1D_rotcurve_components = utils_io.plot_1D_rotcurve_components # before 2025
from dysmalpy.fitting_wrappers.plotting import plot_1D_rotcurve_components # 202509

# <DZLIU><20210726> ++++++++++
from dysmalpy import lensing
from dysmalpy.lensing import LensingTransformer
# <DZLIU><20210726> ----------

# <DZLIU><20210805> ++++++++++
from dysmalpy import utils_least_chi_squares_1d_fitter
from dysmalpy.utils_least_chi_squares_1d_fitter import LeastChiSquares1D
# <DZLIU><20210805> ----------

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import SqrtStretch, AsinhStretch, MinMaxInterval, ImageNormalize

import regions
from regions import (SkyRegion, PixelRegion, PointSkyRegion, PointPixelRegion, CircleSkyRegion, CirclePixelRegion, 
                     EllipseSkyRegion, EllipsePixelRegion, PolygonSkyRegion, PolygonPixelRegion, 
                     RectangleSkyRegion, RectanglePixelRegion, 
                     RegionMask, PixCoord
                    ) # https://astropy-regions.readthedocs.io/en/latest/shapes.html
try:
    # old regions version < 0.6
    from regions import DS9Parser, ds9_objects_to_string
    from regions import read_ds9, write_ds9
except:
    # new regions version >= 0.6
    from regions import Regions
    read_ds9 = lambda x: Regions.read(x, format='ds9')
    write_ds9 = lambda x: Regions.write(x, format='ds9')
    #DS9Parser = lambda x: Regions.parse(x, format='ds9')
    #ds9_objects_to_string =

from spectral_cube import SpectralCube

import matplotlib as mpl
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['legend.borderpad'] = 0.1
mpl.rcParams['legend.labelspacing'] = 0.1
mpl.rcParams['legend.handletextpad'] = 0.1
mpl.rcParams['legend.borderaxespad'] = 0.1
mpl.rcParams['legend.handlelength'] = 1.05
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.framealpha'] = 0.1
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Ellipse, Polygon
from matplotlib.backend_bases import MouseButton
from mpl_toolkits.axes_grid1 import make_axes_locatable

import PyQt5
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtWidgets import (QWidget, QMainWindow, QApplication, QAction, qApp, QTabWidget, QFrame, QSpacerItem,
                             QTabBar, QCheckBox, QMenuBar, QMenu, QToolTip, QLabel, QLineEdit, QTextEdit,
                             QPushButton, QRadioButton, QButtonGroup, QGroupBox, QSplitter,
                             QHBoxLayout, QVBoxLayout, QGridLayout, QFormLayout,
                             QSizePolicy, QFileDialog, QDialog, QDialogButtonBox, QMessageBox,
                             QScrollArea, QComboBox, QLayout,
                             QShortcut)
from PyQt5.QtGui import (QIcon, QFont, QImage, QPixmap, QPolygon, QPolygonF, QPainter, QPen, QTransform,
                         QRegExpValidator, QKeySequence)
from PyQt5.QtCore import (Qt, QByteArray, QObject, QPoint, QPointF, QRect, QRectF, QSize, QSizeF, QThread,
                          QRegExp, QUrl, pyqtSignal, pyqtSlot, pyqtBoundSignal)



#import warnings
#warnings.simplefilter("error")

#galaxy.logger.setLevel(logging.DEBUG)
#models.logger.setLevel(logging.DEBUG)
#fitting.logger.setLevel(logging.DEBUG)
logging.getLogger('DysmalPy').setLevel(logger.level)


#logger.debug('models._dir_noordermeer: '+str(models._dir_noordermeer))


def setup_lensing_dict(params=None, append_to_dict=None):
    lensing_dict = {}
    if append_to_dict is not None:
        if isinstance(append_to_dict, dict):
            lensing_dict = append_to_dict
    for key in params.keys():
        if key.startswith('lensing'):
            lensing_dict[key] = params[key]
    return lensing_dict


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def iconFromBase64(base64_data):
    pixmap = QPixmap()
    pixmap.loadFromData(base64.b64decode(base64_data)) # QByteArray.fromBase64(base64_data)
    icon = QIcon()
    icon.addPixmap(pixmap)
    return icon


def getIconDysmalPy():
    image_base64 = """
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
"""
    # Sedona's art, DPy_favicon_blk-diag-128.png
    return iconFromBase64(image_base64)


class QDysmalPyGUI(QMainWindow):
    """A PyQt5 based DysmalPy GUI."""
    
    def __init__(self, ScreenSize=None):
        #
        super(QDysmalPyGUI, self).__init__()
        #
        self.DysmalPyParamFile = ''
        self.DysmalPyParams = {}
        self.DefaultDirectory = os.getcwd()
        #
        # set self.logger
        self.logger = logging.getLogger('QDysmalPyGUI')
        self.logger.setLevel(logging.getLogger(__name__).level)
        #
        self.logger.debug('proc id: %s, thread: %s, gui'%(str(multiprocessing.current_process().pid),
                                                          str(hex(threading.currentThread().ident))))
        #
        # Create a QDysmalPyFittingTower object that lives in another thread
        # and communicates between the main GUI thread and a QDysmalPyFittingStarship
        # process so that the main GUI thread will not be jammed.
        self.DysmalPyFittingTower = QDysmalPyFittingTower()
        self.DysmalPyFittingStarship = QDysmalPyFittingStarship()
        self.DysmalPyFittingTower.addStarship(self.DysmalPyFittingStarship)
        #
        self.logger.debug('self.initUI()')
        self.initUI(ScreenSize=ScreenSize) # initUI
        #
        self.DysmalPyFittingTower.finished.connect(self.onFittingWorkerFinished)
        self.DysmalPyFittingTower.finishedWithError.connect(self.onFittingWorkerFinishedWithError)
        self.DysmalPyFittingTower.finishedWithWarning.connect(self.onFittingWorkerFinishedWithWarning)
        #
        self.logger.debug('self.DysmalPyFittingTower.start()')
        self.DysmalPyFittingTower.start()
        time.sleep(0.5)
        self.logger.debug('self.DysmalPyFittingStarship.start()')
        self.DysmalPyFittingStarship.start()
        time.sleep(0.5)
        #
        # debug
        #self.selectParamFile('outdir/fitting_1D_mpfit.params')
    
    #def initDysmalPyParams(self):
    #    filepath = 'TODO' #<TODO># a template params file?
    #    self.DysmalPyParams = data_io.read_fitting_params(filepath)
    #    pass
    
    def initUI(self, ScreenSize=None):
        #
        self.CentralWidget = QWidget()
        self.PanelLeft = QWidget()
        self.PanelMiddle = QWidget()
        self.PanelRight = QWidget()
        self.PanelLeft.setMinimumWidth(400)
        self.PanelMiddle.setMinimumWidth(180)
        self.PanelRight.setMinimumWidth(400)
        #
        # Panel Left widgets
        self.LabelForTabWidgetA = QLabel(self.tr('Data'))
        self.LabelForTabWidgetB = QLabel(self.tr('Model'))
        self.TabWidgetHolderA = QWidget()
        self.TabWidgetHolderB = QWidget()
        self.TabWidgetA = QTabWidget()
        self.TabWidgetB = QTabWidget()
        self.TabPageA1 = QWidget()
        self.TabPageA2 = QWidget()
        self.TabPageA3 = QWidget()
        self.TabPageA4 = QWidget()
        self.TabPageA5 = QWidget()
        self.TabPageB1 = QWidget()
        self.TabPageB2 = QWidget()
        self.TabPageB3 = QWidget()
        self.TabPageB4 = QWidget()
        self.TabPageB5 = QWidget()
        self.TabPageB6 = QWidget()
        self.TabPageB7 = QWidget()
        self.TabPageB8 = QWidget()
        self.TabPageA1.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageA2.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageA3.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageA4.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageA5.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageB1.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageB2.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageB3.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageB4.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageB5.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageB6.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageB7.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.TabPageB8.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred) # this is needed otherwise labels will be partially hidden
        self.ScrollAreaForTabPageA1 = QScrollArea(self)
        self.ScrollAreaForTabPageA2 = QScrollArea(self)
        self.ScrollAreaForTabPageA3 = QScrollArea(self)
        self.ScrollAreaForTabPageA4 = QScrollArea(self)
        self.ScrollAreaForTabPageA5 = QScrollArea(self)
        self.ScrollAreaForTabPageB1 = QScrollArea(self)
        self.ScrollAreaForTabPageB2 = QScrollArea(self)
        self.ScrollAreaForTabPageB3 = QScrollArea(self)
        self.ScrollAreaForTabPageB4 = QScrollArea(self)
        self.ScrollAreaForTabPageB5 = QScrollArea(self)
        self.ScrollAreaForTabPageB6 = QScrollArea(self)
        self.ScrollAreaForTabPageB7 = QScrollArea(self)
        self.ScrollAreaForTabPageB8 = QScrollArea(self)
        self.ScrollAreaForTabPageA1.setWidgetResizable(True)
        self.ScrollAreaForTabPageA2.setWidgetResizable(True)
        self.ScrollAreaForTabPageA3.setWidgetResizable(True)
        self.ScrollAreaForTabPageA4.setWidgetResizable(True)
        self.ScrollAreaForTabPageA5.setWidgetResizable(True)
        self.ScrollAreaForTabPageB1.setWidgetResizable(True)
        self.ScrollAreaForTabPageB2.setWidgetResizable(True)
        self.ScrollAreaForTabPageB3.setWidgetResizable(True)
        self.ScrollAreaForTabPageB4.setWidgetResizable(True)
        self.ScrollAreaForTabPageB5.setWidgetResizable(True)
        self.ScrollAreaForTabPageB6.setWidgetResizable(True)
        self.ScrollAreaForTabPageB7.setWidgetResizable(True)
        self.ScrollAreaForTabPageB8.setWidgetResizable(True)
        self.ScrollAreaForTabPageA1.setWidget(self.TabPageA1)
        self.ScrollAreaForTabPageA2.setWidget(self.TabPageA2)
        self.ScrollAreaForTabPageA3.setWidget(self.TabPageA3)
        self.ScrollAreaForTabPageA4.setWidget(self.TabPageA4)
        self.ScrollAreaForTabPageA5.setWidget(self.TabPageA5)
        self.ScrollAreaForTabPageB1.setWidget(self.TabPageB1)
        self.ScrollAreaForTabPageB2.setWidget(self.TabPageB2)
        self.ScrollAreaForTabPageB3.setWidget(self.TabPageB3)
        self.ScrollAreaForTabPageB4.setWidget(self.TabPageB4)
        self.ScrollAreaForTabPageB5.setWidget(self.TabPageB5)
        self.ScrollAreaForTabPageB6.setWidget(self.TabPageB6)
        self.ScrollAreaForTabPageB7.setWidget(self.TabPageB7)
        self.ScrollAreaForTabPageB8.setWidget(self.TabPageB8)
        self.LabelForLineEditParamFile = QLabel(self.tr('Param File:'))
        self.ButtonForLineEditParamFile = QPushButton(self.tr('...'))
        self.ButtonForLineEditParamFile.setToolTip('Open a param file')
        self.ButtonForLineEditParamFile.clicked.connect(self.onOpenParamFileCall)
        self.LineEditParamFile = QLineEdit()
        self.LineEditParamFile.setToolTip(self.tr('Input a Dysmal params file.'))
        #
        self.LineEditDataParamsDict = OrderedDict()
        self.LineEditDataParamsDict['datadir'] = QWidgetForParamInput(\
                    keyname=self.tr('datadir'),
                    keycomment=self.tr('Input data directory.'),
                    datatype=str,
                    fullwidth=True,
                    isdatadir=True, defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata'),
                    keycomment=self.tr('1D data file.'),
                    datatype=str,
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('1D data file (*.txt *.dat *.*)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata_flux'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata_flux'),
                    keycomment=self.tr('2D flux map data file.'),
                    datatype=str,
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('FITS file (*.fits *.fits.gz)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata_ferr'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata_ferr'),
                    keycomment=self.tr('2D flux error map data file.'),
                    datatype=str,
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('FITS file (*.fits *.fits.gz)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata_vel'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata_vel'),
                    keycomment=self.tr('2D velocity map data file.'),
                    datatype=str,
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('FITS file (*.fits *.fits.gz)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata_verr'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata_verr'),
                    keycomment=self.tr('2D velocity map data file.'),
                    datatype=str,
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('FITS file (*.fits *.fits.gz)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata_disp'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata_disp'),
                    keycomment=self.tr('2D dispersion map data file.'),
                    datatype=str,
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('FITS file (*.fits *.fits.gz)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata_derr'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata_derr'),
                    keycomment=self.tr('2D dispersion error map data file.'),
                    datatype=str,
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('FITS file (*.fits *.fits.gz)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata_mask'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata_mask'),
                    keycomment=self.tr('2D or 3D mask data file. 1 for good pixels.'),
                    datatype=str,
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('FITS file (*.fits *.fits.gz)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata_cube'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata_cube'),
                    keycomment=self.tr('3D data cube.'),
                    datatype=str,
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('FITS file (*.fits *.fits.gz)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['fdata_err'] = QWidgetForParamInput(\
                    keyname=self.tr('fdata_err'),
                    keycomment=self.tr('3D data error cube.'),
                    datatype=str,
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('FITS file (*.fits *.fits.gz)'), defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['outdir'] = QWidgetForParamInput(\
                    keyname=self.tr('outdir'),
                    keycomment=self.tr('Output directory.'),
                    datatype=str,
                    fullwidth=True,
                    isoutdir=True, defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditDataParamsDict['galID'] = QWidgetForParamInput(\
                    keyname=self.tr('galID'),
                    keycomment=self.tr('Name of your object'),
                    datatype=str)
        self.LineEditDataParamsDict['z'] = QWidgetForParamInput(\
                    keyname=self.tr('z'),
                    keycomment=self.tr('Redshift'),
                    datatype=float)
        self.LineEditDataParamsDict['data_inst_corr'] = QWidgetForParamInput(\
                    keyname=self.tr('data_inst_corr'),
                    keycomment=self.tr('Is the dispersion corrected for instrumental broadening?'),
                    datatype=bool)
        self.LineEditDataParamsDict['symmetrize_data'] = QWidgetForParamInput(\
                    keyname=self.tr('symmetrize_data'),
                    keycomment=self.tr('Symmetrize data before fitting?'),
                    datatype=bool,
                    default=False)
        self.LineEditDataParamsDict['slit_width'] = QWidgetForParamInput(\
                    keyname=self.tr('slit_width'),
                    keycomment=self.tr('arcsecs'),
                    datatype=float)
        self.LineEditDataParamsDict['slit_pa'] = QWidgetForParamInput(\
                    keyname=self.tr('slit_pa'),
                    keycomment=self.tr('Degrees from N towards blue'),
                    datatype=float)
        self.LineEditDataParamsDict['profile1d_type'] = QWidgetForParamInput(\
                    keyname=self.tr('profile1d_type'),
                    keycomment=self.tr('Default 1D aperture extraction shape'),
                    datatype=str,
                    options=['circ_ap_cube', 'rect_ap_cube', 'square_ap_cube', 'circ_ap_pv', 'single_pix_pv'],
                    default='rect_ap_cube')
        self.LineEditDataParamsDict['aperture_radius'] = QWidgetForParamInput(\
                    keyname=self.tr('aperture_radius'),
                    keycomment=self.tr('Circular aperture radius, in ARCSEC. Have used half slit width in past'),
                    datatype=float,
                    default=0.2)
        self.LineEditDataParamsDict['pix_parallel'] = QWidgetForParamInput(\
                    keyname=self.tr('pix_parallel'),
                    keycomment=self.tr('Rectangle aperture size in parallel direction of the pseudo slit, in pixels.'),
                    datatype=float,
                    default=1)
        self.LineEditDataParamsDict['pix_perp'] = QWidgetForParamInput(\
                    keyname=self.tr('pix_perp'),
                    keycomment=self.tr('Rectangle aperture size in perpendicular direction of the pseudo slit, i.e., slit width, in pixels.'),
                    datatype=float,
                    default=10.)
        self.LineEditDataParamsDict['smoothing_type'] = QWidgetForParamInput(\
                    keyname=self.tr('smoothing_type'),
                    keycomment=self.tr('Is the data median smoothed before extracting maps?'),
                    datatype=str,
                    options=['None', 'median'])
        self.LineEditDataParamsDict['smoothing_npix'] = QWidgetForParamInput(\
                    keyname=self.tr('smoothing_npix'),
                    keycomment=self.tr('Number of pixels for smoothing aperture'),
                    datatype=float,
                    default=1.0)
        self.LineEditDataParamsDict['xcenter'] = QWidgetForParamInput(\
                    keyname=self.tr('xcenter'),
                    keycomment=self.tr('Galaxy center in pixel coordinate, starting from 0 to NX-1. Need +1 for QFitsView. None means using image center.'),
                    datatype=float,
                    default='None')
        self.LineEditDataParamsDict['ycenter'] = QWidgetForParamInput(\
                    keyname=self.tr('ycenter'),
                    keycomment=self.tr('Galaxy center in pixel coordinate, starting from 0 to NY-1. Need +1 for QFitsView. None means using image center.'),
                    datatype=float,
                    default='None')
        self.LineEditDataParamsDict['linked_posteriors'] = QWidgetForParamInput(\
                    keyname=self.tr('linked_posteriors'),
                    keycomment=self.tr(''),
                    datatype=str,
                    listtype=list,
                    default="['total_mass', 'r_eff_disk', 'bt', 'fdm', 'sigma0']",
                    fullwidth=True)
        self.LineEditDataParamsDict['pixscale'] = QWidgetForParamInput(\
                    keyname=self.tr('pixscale'),
                    keycomment=self.tr('Pixel scale in arcsec/pixel'),
                    datatype=float)
        self.LineEditDataParamsDict['fov_npix'] = QWidgetForParamInput(\
                    keyname=self.tr('fov_npix'),
                    keycomment=self.tr('Number of pixels on a side of model cube'),
                    datatype=int)
        self.LineEditDataParamsDict['spec_type'] = QWidgetForParamInput(\
                    keyname=self.tr('spec_type'),
                    keycomment=self.tr('Spectral type, must be velocity.'),
                    datatype=str,
                    options=['velocity'])
        self.LineEditDataParamsDict['spec_start'] = QWidgetForParamInput(\
                    keyname=self.tr('spec_start'),
                    keycomment=self.tr('Starting value for spectral axis'),
                    datatype=float)
        self.LineEditDataParamsDict['spec_step'] = QWidgetForParamInput(\
                    keyname=self.tr('spec_step'),
                    keycomment=self.tr('Step size for spectral axis in km/s'),
                    datatype=float)
        self.LineEditDataParamsDict['nspec'] = QWidgetForParamInput(\
                    keyname=self.tr('nspec'),
                    keycomment=self.tr('Number of spectral steps'),
                    datatype=int)
        self.LineEditDataParamsDict['use_lsf'] = QWidgetForParamInput(\
                    keyname=self.tr('use_lsf'),
                    keycomment=self.tr('True/False if using an LSF'),
                    datatype=bool)
        self.LineEditDataParamsDict['sig_inst_res'] = QWidgetForParamInput(\
                    keyname=self.tr('sig_inst_res'),
                    keycomment=self.tr('Instrumental dispersion in km/s'),
                    datatype=float)
        self.LineEditDataParamsDict['psf_type'] = QWidgetForParamInput(\
                    keyname=self.tr('psf_type'),
                    keycomment=self.tr('PSF type, Gaussian or Moffat or DoubleGaussian.'),
                    datatype=str,
                    options=['Gaussian', 'Moffat', 'DoubleGaussian'])
        self.LineEditDataParamsDict['psf_fwhm'] = QWidgetForParamInput(\
                    keyname=self.tr('psf_fwhm'),
                    keycomment=self.tr('PSF FWHM in arcsecs'),
                    datatype=float)
        self.LineEditDataParamsDict['psf_beta'] = QWidgetForParamInput(\
                    keyname=self.tr('psf_beta'),
                    keycomment=self.tr('Beta parameter for a Moffat PSF'),
                    datatype=float)
        self.LineEditDataParamsDict['psf_fwhm_major'] = QWidgetForParamInput( \
                    keyname=self.tr('psf_fwhm_major'),
                    keycomment=self.tr('If using an elliptical PSF, set here the PSF major axis FWHM in arcsecs'),
                    datatype=float)
        self.LineEditDataParamsDict['psf_fwhm_minor'] = QWidgetForParamInput( \
                    keyname=self.tr('psf_fwhm_minor'),
                    keycomment=self.tr('If using an elliptical PSF, set here the PSF minor axis FWHM in arcsecs'),
                    datatype=float)
        self.LineEditDataParamsDict['psf_PA'] = QWidgetForParamInput( \
                    keyname=self.tr('psf_PA'),
                    keycomment=self.tr('If using an elliptical PSF, set here the PSF position angle in degrees'),
                    datatype=float)
        #
        #self.LineEditInstrumentParamsDict = OrderedDict()
        #self.LineEditInstrumentParamsDict['datadir'] = QWidgetForParamInput(\
        #            TODO
        #
        self.LineEditLensingParamsDict = OrderedDict()
        self.LineEditLensingParamsDict['lensing_datadir'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_datadir'),
                    keycomment=self.tr('Glafic lensing model mesh.dat directory.'),
                    datatype=str,
                    fullwidth=True,
                    isdatadir=True, defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditLensingParamsDict['lensing_mesh'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_mesh'),
                    keycomment=self.tr('Glafic lensing model mesh.dat file.'),
                    datatype=str,
                    fullwidth=True,
                    isdatafile=True, namefilter=self.tr('Glafic lensing model mesh.dat file (*.dat)'),
                    defaultdir=self.DefaultDirectory, enabled=False)
        self.LineEditLensingParamsDict['lensing_ra'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_ra'),
                    keycomment=self.tr('Lensing model reference WCS RA.'),
                    datatype=float)
        self.LineEditLensingParamsDict['lensing_dec'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_dec'),
                    keycomment=self.tr('Lensing model reference WCS Dec.'),
                    datatype=float)
        self.LineEditLensingParamsDict['lensing_sra'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_sra'),
                    keycomment=self.tr('Source plane map center RA.'),
                    datatype=float)
        self.LineEditLensingParamsDict['lensing_sdec'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_sdec'),
                    keycomment=self.tr('Source plane map center Dec.'),
                    datatype=float)
        self.LineEditLensingParamsDict['lensing_ssizex'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_ssizex'),
                    keycomment=self.tr('Source plane map size in pixels.'),
                    datatype=float)
        self.LineEditLensingParamsDict['lensing_ssizey'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_ssizey'),
                    keycomment=self.tr('Source plane map size in pixels.'),
                    datatype=float)
        self.LineEditLensingParamsDict['lensing_spixsc'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_spixsc'),
                    keycomment=self.tr('Source plane map pixel size in units of arcsec.'),
                    datatype=float)
        self.LineEditLensingParamsDict['lensing_imra'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_imra'),
                    keycomment=self.tr('Image plane map center RA.'),
                    datatype=float)
        self.LineEditLensingParamsDict['lensing_imdec'] = QWidgetForParamInput(\
                    keyname=self.tr('lensing_imdec'),
                    keycomment=self.tr('Image plane map center Dec.'),
                    datatype=float)
        #
        self.LineEditModelParamsDictForBulgeDisk = OrderedDict()
        self.LineEditModelParamsDictForBulgeDisk['total_mass'] = QWidgetForParamInput(\
                    keyname=self.tr('total_mass'),
                    keycomment=self.tr("Total mass of disk and bulge log(Msun)"),
                    datatype=float)
        self.LineEditModelParamsDictForBulgeDisk['total_mass_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('total_mass_fixed'),
                    keycomment=self.tr(""),
                    datatype=bool)
        self.LineEditModelParamsDictForBulgeDisk['bt'] = QWidgetForParamInput(\
                    keyname=self.tr('bt'),
                    keycomment=self.tr("Bulge-to-Total Ratio"),
                    datatype=float)
        self.LineEditModelParamsDictForBulgeDisk['bt_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr("bt_fixed"),
                    keycomment=self.tr("Fix bulge-to-total ratio?"),
                    datatype=bool)
        self.LineEditModelParamsDictForBulgeDisk['r_eff_disk'] = QWidgetForParamInput(\
                    keyname=self.tr('r_eff_disk'),
                    keycomment=self.tr("Effective radius of disk in kpc"),
                    datatype=float)
        self.LineEditModelParamsDictForBulgeDisk['r_eff_disk_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr("r_eff_disk_fixed"),
                    keycomment=self.tr("Fix R_eff?"),
                    datatype=bool)
        self.LineEditModelParamsDictForBulgeDisk['n_disk'] = QWidgetForParamInput(\
                    keyname=self.tr('n_disk'),
                    keycomment=self.tr("Sersic index for disk"),
                    datatype=float)
        self.LineEditModelParamsDictForBulgeDisk['n_disk_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr("n_disk_fixed"),
                    keycomment=self.tr("Fix n_disk?"),
                    datatype=bool)
        self.LineEditModelParamsDictForBulgeDisk['invq_disk'] = QWidgetForParamInput(\
                    keyname=self.tr('invq_disk'),
                    keycomment=self.tr("disk scale length to zheight ratio for disk"),
                    datatype=float,
                    default=5.0)
        self.LineEditModelParamsDictForBulgeDisk['invq_disk_NULL'] = QWidgetForParamInput(\
                    keyname=self.tr('NULL'),
                    keycomment=self.tr(""))
        self.LineEditModelParamsDictForBulgeDisk['r_eff_bulge'] = QWidgetForParamInput(\
                    keyname=self.tr('r_eff_bulge'),
                    keycomment=self.tr("Effective radius of bulge in kpc"),
                    datatype=float)
        self.LineEditModelParamsDictForBulgeDisk['r_eff_bulge_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr("r_eff_bulge_fixed"),
                    keycomment=self.tr("Fix R_eff and bulge-to-total?"),
                    datatype=bool)
        self.LineEditModelParamsDictForBulgeDisk['n_bulge'] = QWidgetForParamInput(\
                    keyname=self.tr('n_bulge'),
                    keycomment=self.tr("Sersic index for bulge"),
                    datatype=float)
        self.LineEditModelParamsDictForBulgeDisk['n_bulge_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr("n_bulge_fixed"),
                    keycomment=self.tr("Fix n_disk and bulge-to-total?"),
                    datatype=bool)
        self.LineEditModelParamsDictForBulgeDisk['invq_bulge'] = QWidgetForParamInput(\
                    keyname=self.tr('invq_bulge'),
                    keycomment=self.tr("disk scale length to zheight ratio for bulge"),
                    datatype=float,
                    default=2.0)
        self.LineEditModelParamsDictForBulgeDisk['invq_bulge_NULL'] = QWidgetForParamInput(\
                    keyname=self.tr('NULL'),
                    keycomment=self.tr(""))
        #
        self.LineEditModelParamsDictForDarkMatterHalo = OrderedDict()
        self.LineEditModelParamsDictForDarkMatterHalo['include_halo'] = QWidgetForParamInput(\
                    keyname=self.tr('include_halo'),
                    keycomment=self.tr("Include a halo?"),
                    #associatedparams=['halo_profile_type', 'mvirial', 'mvirial_fixed', 'halo_conc', 'halo_conc_fixed', 'fdm', 'fdm_fixed'],
                    datatype=bool,
                    default=True)
        self.LineEditModelParamsDictForDarkMatterHalo['include_halo_NULL'] = QWidgetForParamInput(\
                    keyname=self.tr('NULL'),
                    keycomment=self.tr(""))
        self.LineEditModelParamsDictForDarkMatterHalo['halo_profile_type'] = QWidgetForParamInput(\
                    keyname=self.tr('halo_profile_type'),
                    keycomment=self.tr("Halo type"),
                    datatype=str,
                    options=['NFW', 'twopowerhalo', 'burkert', 'einasto', 'dekelzhao'])
        self.LineEditModelParamsDictForDarkMatterHalo['halo_profile_type_NULL'] = QWidgetForParamInput(\
                    keyname=self.tr('NULL'),
                    keycomment=self.tr(""))
        self.LineEditModelParamsDictForDarkMatterHalo['mvirial'] = QWidgetForParamInput(\
                    keyname=self.tr('mvirial'),
                    keycomment=self.tr("Halo virial mass in log(Msun)"),
                    datatype=float)
        self.LineEditModelParamsDictForDarkMatterHalo['mvirial_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('mvirial_fixed'),
                    keycomment=self.tr(""),
                    datatype=bool)
        self.LineEditModelParamsDictForDarkMatterHalo['halo_conc'] = QWidgetForParamInput(\
                    keyname=self.tr('halo_conc'),
                    keycomment=self.tr("Halo concentration parameter"),
                    datatype=float)
        self.LineEditModelParamsDictForDarkMatterHalo['halo_conc_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('halo_conc_fixed'),
                    keycomment=self.tr(""),
                    datatype=bool)
        self.LineEditModelParamsDictForDarkMatterHalo['alpha'] = QWidgetForParamInput(\
                    keyname=self.tr('alpha'),
                    keycomment=self.tr("Halo alpha parameter for the twopowerhalo type, default is 1."),
                    datatype=float, default=1.0)
        self.LineEditModelParamsDictForDarkMatterHalo['alpha_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('alpha_fixed'),
                    keycomment=self.tr(""),
                    datatype=bool)
        self.LineEditModelParamsDictForDarkMatterHalo['beta'] = QWidgetForParamInput(\
                    keyname=self.tr('beta'),
                    keycomment=self.tr("Halo beta parameter for the twopowerhalo type, default is 3."),
                    datatype=float, default=3.0)
        self.LineEditModelParamsDictForDarkMatterHalo['beta_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('beta_fixed'),
                    keycomment=self.tr(""),
                    datatype=bool)
        self.LineEditModelParamsDictForDarkMatterHalo['fdm'] = QWidgetForParamInput(\
                    keyname=self.tr('fdm'),
                    keycomment=self.tr("Dark matter fraction at Reff"),
                    datatype=float)
        self.LineEditModelParamsDictForDarkMatterHalo['fdm_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('fdm_fixed'),
                    keycomment=self.tr(""),
                    datatype=bool)
        #
        self.LineEditModelParamsDictForDispersion = OrderedDict()
        self.LineEditModelParamsDictForDispersion['sigma0'] = QWidgetForParamInput(\
                    keyname=self.tr('sigma0'),
                    keycomment=self.tr("Constant intrinsic dispersion value"),
                    datatype=float)
        self.LineEditModelParamsDictForDispersion['sigma0_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('sigma0_fixed'),
                    keycomment=self.tr(""),
                    datatype=bool)
        #
        self.LineEditModelParamsDictForGeometry = OrderedDict()
        self.LineEditModelParamsDictForGeometry['inc'] = QWidgetForParamInput(\
                    keyname=self.tr('inc'),
                    keycomment=self.tr("Inclination of galaxy, 0=face-on, 90=edge-on"),
                    datatype=float)
        self.LineEditModelParamsDictForGeometry['inc_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('inc_fixed'),
                    keycomment=self.tr(""),
                    datatype=bool)
        self.LineEditModelParamsDictForGeometry['pa'] = QWidgetForParamInput(\
                    keyname=self.tr('pa'),
                    keycomment=self.tr("Position angle of galaxy."),
                    datatype=float)
        self.LineEditModelParamsDictForGeometry['pa_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('pa_fixed'),
                    keycomment=self.tr(""),
                    datatype=bool)
        self.LineEditModelParamsDictForGeometry['xshift'] = QWidgetForParamInput(\
                    keyname=self.tr('xshift'),
                    keycomment=self.tr("xshift in pixels"),
                    datatype=float,
                    default=0.0)
        self.LineEditModelParamsDictForGeometry['xshift_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('xshift_fixed'),
                    keycomment=self.tr(""),
                    datatype=bool)
        self.LineEditModelParamsDictForGeometry['yshift'] = QWidgetForParamInput(\
                    keyname=self.tr('yshift'),
                    keycomment=self.tr("yshift in pixels"),
                    datatype=float,
                    default=0.0)
        self.LineEditModelParamsDictForGeometry['yshift_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('yshift_fixed'),
                    keycomment=self.tr(""),
                    datatype=bool)
        self.LineEditModelParamsDictForGeometry['vel_shift'] = QWidgetForParamInput(\
                    keyname=self.tr('vel_shift'),
                    keycomment=self.tr("vel_shift in km/s"),
                    datatype=float,
                    default=0.0)
        self.LineEditModelParamsDictForGeometry['vel_shift_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('vel_shift_fixed'),
                    keycomment=self.tr(""),
                    datatype=bool)
        #
        self.LineEditModelParamsDictForZHeight = OrderedDict()
        self.LineEditModelParamsDictForZHeight['sigmaz'] = QWidgetForParamInput(\
                    keyname=self.tr('sigmaz'),
                    keycomment=self.tr("Gaussian width of the galaxy in z-direction"),
                    datatype=float)
        self.LineEditModelParamsDictForZHeight['sigmaz_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('sigmaz_fixed'),
                    keycomment=self.tr(""),
                    datatype=bool)
        #
        self.LineEditModelParamsDictForNonCirc = OrderedDict()
        self.LineEditModelParamsDictForNonCirc['include_uniform_planar_radial_flow'] = QWidgetForParamInput(\
                    keyname=self.tr('include_uniform_planar_radial_flow'),
                    keycomment=self.tr("enable uniform_planar_radial_flow?"),
                    associatedparams=['vr', 'vr_fixed'],
                    datatype=bool, 
                    default=False)
        self.LineEditModelParamsDictForNonCirc['include_uniform_planar_radial_flow_NULL'] = QWidgetForParamInput(\
                    keyname=self.tr('NULL'),
                    keycomment=self.tr(""))
        self.LineEditModelParamsDictForNonCirc['vr'] = QWidgetForParamInput(\
                    keyname=self.tr('vr'),
                    keycomment=self.tr("Radial flow velocity [km/s]. Positive: Outflow. Negative: Inflow."),
                    datatype=float, 
                    default=30.0, enabled=False)
        self.LineEditModelParamsDictForNonCirc['vr_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('vr_fixed'),
                    keycomment=self.tr(""),
                    datatype=bool, 
                    default=False, enabled=False)
        self.LineEditModelParamsDictForNonCirc['include_uniform_bar_flow'] = QWidgetForParamInput(\
                    keyname=self.tr('include_uniform_bar_flow'),
                    keycomment=self.tr("enable uniform_bar_flow?"),
                    associatedparams=['vbar', 'vbar_fixed', 'phi', 'phi_fixed', 'bar_width', 'bar_width_fixed'],
                    datatype=bool, 
                    default=False)
        self.LineEditModelParamsDictForNonCirc['include_uniform_bar_flow_NULL'] = QWidgetForParamInput(\
                    keyname=self.tr('NULL'),
                    keycomment=self.tr(""))
        self.LineEditModelParamsDictForNonCirc['vbar'] = QWidgetForParamInput(\
                    keyname=self.tr('vbar'),
                    keycomment=self.tr("Bar flow velocity [km/s]. Positive: Outflow. Negative: Inflow."),
                    datatype=float, 
                    default=-90.0, enabled=False)
        self.LineEditModelParamsDictForNonCirc['vbar_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('vbar_fixed'),
                    keycomment=self.tr(""),
                    datatype=bool, 
                    default=False, enabled=False)
        self.LineEditModelParamsDictForNonCirc['phi'] = QWidgetForParamInput(\
                    keyname=self.tr('phi'),
                    keycomment=self.tr("Azimuthal angle of bar [degrees], counter-clockwise from blue major axis. Default is 90 (eg, along galaxy minor axis)."),
                    datatype=float, 
                    default=90.0, enabled=False)
        self.LineEditModelParamsDictForNonCirc['phi_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('phi_fixed'),
                    keycomment=self.tr(""),
                    datatype=bool, 
                    default=True, enabled=False) # in default this is fixed
        self.LineEditModelParamsDictForNonCirc['bar_width'] = QWidgetForParamInput(\
                    keyname=self.tr('bar_width'),
                    keycomment=self.tr("Width of the bar perpendicular to bar direction, in pixels. Bar velocity only is nonzero between -bar_width/2 < ygal < bar_width/2."),
                    datatype=float, 
                    default=5.0, enabled=False)
        self.LineEditModelParamsDictForNonCirc['bar_width_fixed'] = QWidgetForParamInput(\
                    keyname=self.tr('bar_width_fixed'),
                    keycomment=self.tr(""),
                    datatype=bool, 
                    default=True, enabled=False) # in default this is fixed
        #
        initial_param_bounds = {}
        initial_param_bounds['total_mass']   = [9.0, 13.0]
        initial_param_bounds['bt']           = [0.0, 1.0]
        initial_param_bounds['r_eff_disk']   = [0.1, 30.0]
        initial_param_bounds['n_disk']       = [1.0, 8.0]
        initial_param_bounds['r_eff_bulge']  = [1.0, 5.0]
        initial_param_bounds['n_bulge']      = [1.0, 8.0]
        initial_param_bounds['sigma0']       = [5.0, 300.0]
        initial_param_bounds['sigmaz']       = [0.1, 1.0]
        initial_param_bounds['inc']          = [0.0, 90.0]
        initial_param_bounds['pa']           = [-180.0, 360.0] # pa_bounds
        initial_param_bounds['mvirial']      = [9.0, 14.0]
        initial_param_bounds['halo_conc']    = [1.0, 20.0]
        initial_param_bounds['alpha']        = [0.0, 1.0]
        initial_param_bounds['beta']         = [2.0, 3.0]
        initial_param_bounds['fdm']          = [0.0, 0.99]
        initial_param_bounds['xshift']       = [-10.0, 10.0]
        initial_param_bounds['yshift']       = [-10.0, 10.0]
        initial_param_bounds['vel_shift']    = [-50.0, 50.0]
        initial_param_stddev = {}
        initial_param_stddev['total_mass']   = 1.0
        initial_param_stddev['bt']           = 0.1
        initial_param_stddev['r_eff_disk']   = 1.0
        initial_param_stddev['n_disk']       = 0.1
        initial_param_stddev['r_eff_bulge']  = 1.0
        initial_param_stddev['n_bulge']      = 0.1
        initial_param_stddev['sigma0']       = 25.0
        initial_param_stddev['sigmaz']       = 0.1
        initial_param_stddev['inc']          = 0.1
        initial_param_stddev['pa']           = 0.1
        initial_param_stddev['mvirial']      = 0.5
        initial_param_stddev['halo_conc']    = 0.5
        initial_param_stddev['alpha']        = 0.1
        initial_param_stddev['beta']         = 0.1
        initial_param_stddev['fdm']          = 1.0
        initial_param_stddev['xshift']       = 3.0
        initial_param_stddev['yshift']       = 3.0
        initial_param_stddev['vel_shift']    = 10.0
        #
        self.LineEditModelParamsDictForLimits = OrderedDict()
        for key in ['total_mass', 'bt', 'r_eff_disk', 'n_disk', 'r_eff_bulge', 'n_bulge',
                    'sigma0', 'sigmaz', 'inc', 'pa',
                    'mvirial', 'halo_conc', 'alpha', 'beta', 'fdm', 
                    'xshift', 'yshift', 'vel_shift']:
            self.LineEditModelParamsDictForLimits[key+'_bounds'] = QWidgetForParamInput(\
                    keyname=self.tr(key+'_bounds'),
                    keycomment=self.tr(""),
                    datatype=float,
                    listtype=list,
                    default=str(initial_param_bounds[key]),
                    fullwidth=True)
            self.LineEditModelParamsDictForLimits[key+'_prior'] = QWidgetForParamInput(\
                    keyname=self.tr(key+'_prior'),
                    keycomment=self.tr(""),
                    datatype=str,
                    options=['flat','gaussian'])
            self.LineEditModelParamsDictForLimits[key+'_stddev'] = QWidgetForParamInput(\
                    keyname=self.tr(key+'_stddev'),
                    keycomment=self.tr(""),
                    datatype=float,
                    default=initial_param_stddev[key])
        #
        self.LineEditModelParamsDictForFitting = OrderedDict()
        self.LineEditModelParamsDictForFitting['fit_method'] = QWidgetForParamInput(\
                    keyname=self.tr('fit_method'),
                    keycomment=self.tr("mcmc or mpfit"),
                    datatype=str,
                    options=['mpfit', 'mcmc'])
        self.LineEditModelParamsDictForFitting['moment_calc'] = QWidgetForParamInput(\
                    keyname=self.tr('moment_calc'),
                    keycomment=self.tr('If False, observed maps fit with GAUSSIANS'),
                    datatype=bool,
                    default=False)
        self.LineEditModelParamsDictForFitting['fitdispersion'] = QWidgetForParamInput(\
                    keyname=self.tr('fitdispersion'),
                    keycomment=self.tr("Simultaneously fit the velocity and dispersion?"),
                    datatype=bool)
        self.LineEditModelParamsDictForFitting['fitflux'] = QWidgetForParamInput(\
                    keyname=self.tr('fitflux'),
                    keycomment=self.tr("Simultaneously fit the flux?"),
                    datatype=bool,
                    default=False)
        self.LineEditModelParamsDictForFitting['do_plotting'] = QWidgetForParamInput(\
                    keyname=self.tr('do_plotting'),
                    keycomment=self.tr("Produce all output plots?"),
                    datatype=bool)
        self.LineEditModelParamsDictForFitting['do_plotting_NULL'] = QWidgetForParamInput(\
                    keyname=self.tr('do_plotting_NULL'),
                    keycomment=self.tr("NULL"),
                    datatype=bool,
                    default=False)
        self.LineEditModelParamsDictForFitting['oversample'] = QWidgetForParamInput(\
                    keyname=self.tr('oversample'),
                    keycomment=self.tr("Oversampling the model cube"),
                    datatype=int,
                    default=1)
        self.LineEditModelParamsDictForFitting['oversize'] = QWidgetForParamInput(\
                    keyname=self.tr('oversize'),
                    keycomment=self.tr("Oversize of the model cube"),
                    datatype=int,
                    default=1)
        self.LineEditModelParamsDictForFitting['nWalkers'] = QWidgetForParamInput(\
                    keyname=self.tr('nWalkers'),
                    keycomment=self.tr("Number of walkers. Must be even and >= 2x the number of free parameters."),
                    datatype=int,
                    default=10)
        self.LineEditModelParamsDictForFitting['nCPUs'] = QWidgetForParamInput(\
                    keyname=self.tr('nCPUs'),
                    keycomment=self.tr("Number of CPUs to use for parallelization"),
                    datatype=int,
                    default=1)
        self.LineEditModelParamsDictForFitting['nBurn'] = QWidgetForParamInput(\
                    keyname=self.tr('nBurn'),
                    keycomment=self.tr("Number of steps during burn-in"),
                    datatype=int,
                    default=1)
        self.LineEditModelParamsDictForFitting['nSteps'] = QWidgetForParamInput(\
                    keyname=self.tr('nSteps'),
                    keycomment=self.tr("Number of steps for sampling"),
                    datatype=int,
                    default=100)
        self.LineEditModelParamsDictForFitting['scale_param_a'] = QWidgetForParamInput(\
                    keyname=self.tr('scale_param_a'),
                    keycomment=self.tr(""),
                    datatype=float,
                    default=3.0)
        self.LineEditModelParamsDictForFitting['minAF'] = QWidgetForParamInput(\
                    keyname=self.tr('minAF'),
                    keycomment=self.tr("Minimum acceptance fraction"),
                    datatype=float,
                    default='None')
        self.LineEditModelParamsDictForFitting['maxAF'] = QWidgetForParamInput(\
                    keyname=self.tr('maxAF'),
                    keycomment=self.tr("Maximum acceptance fraction"),
                    datatype=float,
                    default='None')
        self.LineEditModelParamsDictForFitting['nEff'] = QWidgetForParamInput(\
                    keyname=self.tr('nEff'),
                    keycomment=self.tr("Number of auto-correlation times before convergence"),
                    datatype=int,
                    default=10)
        self.LineEditModelParamsDictForFitting['maxiter'] = QWidgetForParamInput(\
                    keyname=self.tr('maxiter'),
                    keycomment=self.tr("Maximum number of iterations before mpfit quits"),
                    datatype=int,
                    default=200)
        self.LineEditModelParamsDictForFitting['overwrite'] = QWidgetForParamInput(\
                    keyname=self.tr('overwrite'),
                    keycomment=self.tr("Overwrite outputs"),
                    datatype=bool,
                    default='False')
        #
        self.LineEditModelParamsDicts = [] # each element is a dict corresponding to a 'self.TabPageB*'
        self.LineEditModelParamsDicts.append(self.LineEditModelParamsDictForBulgeDisk)
        self.LineEditModelParamsDicts.append(self.LineEditModelParamsDictForDarkMatterHalo)
        self.LineEditModelParamsDicts.append(self.LineEditModelParamsDictForDispersion)
        self.LineEditModelParamsDicts.append(self.LineEditModelParamsDictForGeometry)
        self.LineEditModelParamsDicts.append(self.LineEditModelParamsDictForZHeight)
        self.LineEditModelParamsDicts.append(self.LineEditModelParamsDictForNonCirc)
        self.LineEditModelParamsDicts.append(self.LineEditModelParamsDictForLimits)
        self.LineEditModelParamsDicts.append(self.LineEditModelParamsDictForFitting)
        #
        self.TabWidgetA.addTab(self.ScrollAreaForTabPageA1, self.tr('Data'))
        self.TabWidgetA.addTab(self.ScrollAreaForTabPageA2, self.tr('Data 2D'))
        self.TabWidgetA.addTab(self.ScrollAreaForTabPageA3, self.tr('Data 3D'))
        self.TabWidgetA.addTab(self.ScrollAreaForTabPageA4, self.tr('Instrument'))
        self.TabWidgetA.addTab(self.ScrollAreaForTabPageA5, self.tr('Lensing'))
        self.TabWidgetA.setTabToolTip(0, self.tr('Data setup'))
        self.TabWidgetA.setTabToolTip(1, self.tr('Instrument setup'))
        self.TabWidgetA.setTabToolTip(2, self.tr('Lensing transformation'))
        #
        self.TabWidgetB.addTab(self.ScrollAreaForTabPageB1, self.tr('Disk+Bulge'))
        self.TabWidgetB.addTab(self.ScrollAreaForTabPageB2, self.tr('Halo'))
        self.TabWidgetB.addTab(self.ScrollAreaForTabPageB3, self.tr('Disp'))
        self.TabWidgetB.addTab(self.ScrollAreaForTabPageB4, self.tr('Geometry'))
        self.TabWidgetB.addTab(self.ScrollAreaForTabPageB5, self.tr('ZHeight'))
        self.TabWidgetB.addTab(self.ScrollAreaForTabPageB6, self.tr('NonCirc'))
        self.TabWidgetB.addTab(self.ScrollAreaForTabPageB7, self.tr('Limits'))
        self.TabWidgetB.addTab(self.ScrollAreaForTabPageB8, self.tr('Fitting'))
        self.TabWidgetB.setTabToolTip(0, self.tr('Disk+Bulge model parameters'))
        self.TabWidgetB.setTabToolTip(1, self.tr('Halo model parameters'))
        self.TabWidgetB.setTabToolTip(2, self.tr('Dispersion profile parameters'))
        self.TabWidgetB.setTabToolTip(3, self.tr('Geometric parameters'))
        self.TabWidgetB.setTabToolTip(4, self.tr('Z-direction parameters'))
        self.TabWidgetB.setTabToolTip(5, self.tr('Non-circular motion parameters'))
        self.TabWidgetB.setTabToolTip(6, self.tr('All parameter limits'))
        self.TabWidgetB.setTabToolTip(7, self.tr('Parameters related to the fitting'))
        #
        self.TabWidgetA.setStyleSheet("""
            QTabWidget::tab-bar {
                left: 0;
            }""")
        self.TabWidgetB.setStyleSheet("""
            QTabWidget::tab-bar {
                left: 0;
            }""")
        #self.PanelSpecA = QWidget()
        #self.PanelSpecB = QWidget()
        #
        for keyname in ['fdata', 'fdata_flux', 'fdata_ferr', 'fdata_vel', 'fdata_verr', 'fdata_disp', 'fdata_derr', 'fdata_mask', 'fdata_cube', 'fdata_err']:
            self.LineEditDataParamsDict['datadir'].ParamUpdateSignal.connect(self.LineEditDataParamsDict[keyname].onDataDirParamUpdateCall)
        #
        # Panel Left layout
        self.LayoutForLineEditParamFile = QHBoxLayout()
        self.LayoutForLineEditParamFile.setContentsMargins(0, 0, 0, 0)
        self.LayoutForLineEditParamFile.setSpacing(0)
        self.LayoutForLineEditParamFile.addWidget(self.LabelForLineEditParamFile)
        self.LayoutForLineEditParamFile.addWidget(self.LineEditParamFile)
        self.LayoutForLineEditParamFile.addWidget(self.ButtonForLineEditParamFile)
        self.LayoutForLineEditDataParams = QGridLayout()
        self.LayoutForLineEditDataParams.setHorizontalSpacing(30)
        self.LayoutForLineEditDataParams.setVerticalSpacing(5)
        ncolumn = 2
        icount = 0 # self.LayoutForLineEditDataParams.count() * ncolumn # the widgets above are full width
        for key in self.LineEditDataParamsDict:
            if not (key.find('NULL')>=0):
                self.LineEditDataParamsDict[key].ParamUpdateSignal.connect(self.onParamUpdateCall)
                if self.LineEditDataParamsDict[key].isEnabled():
                    self.DysmalPyParams[key] = self.LineEditDataParamsDict[key].keyvalue()
                icount, irow, icol, rowSpan, colSpan = self.LineEditDataParamsDict[key].getPositionInQGridLayout(icount, ncolumn)
                self.LayoutForLineEditDataParams.addWidget(self.LineEditDataParamsDict[key], irow, icol, rowSpan, colSpan)
            else:
                icount += 1
        self.LayoutForTabPageA1 = QVBoxLayout()
        self.LayoutForTabPageA1.setContentsMargins(0, 0, 0, 0)
        self.LayoutForTabPageA1.setSpacing(0)
        self.LayoutForTabPageA1.addItem(self.LayoutForLineEditParamFile)
        self.LayoutForTabPageA1.addItem(self.LayoutForLineEditDataParams)
        self.LayoutForTabPageA1.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.TabPageA1.setLayout(self.LayoutForTabPageA1)
        #
        self.LayoutForLineEditLensingParams = QGridLayout()
        self.LayoutForLineEditLensingParams.setHorizontalSpacing(30)
        self.LayoutForLineEditLensingParams.setVerticalSpacing(5)
        ncolumn = 2
        icount = 0 # self.LayoutForLineEditLensingParams.count() * ncolumn # the widgets above are full width
        for key in self.LineEditLensingParamsDict:
            if not (key.find('NULL')>=0):
                self.LineEditLensingParamsDict[key].ParamUpdateSignal.connect(self.onParamUpdateCall)
                if self.LineEditLensingParamsDict[key].isEnabled():
                    self.DysmalPyParams[key] = self.LineEditLensingParamsDict[key].keyvalue()
                icount, irow, icol, rowSpan, colSpan = self.LineEditLensingParamsDict[key].getPositionInQGridLayout(icount, ncolumn)
                self.LayoutForLineEditLensingParams.addWidget(self.LineEditLensingParamsDict[key], irow, icol, rowSpan, colSpan)
            else:
                icount += 1
        self.LayoutForTabPageA5 = QVBoxLayout()
        self.LayoutForTabPageA5.setContentsMargins(0, 0, 0, 0)
        self.LayoutForTabPageA5.setSpacing(0)
        self.LayoutForTabPageA5.addItem(self.LayoutForLineEditLensingParams)
        self.LayoutForTabPageA5.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.TabPageA5.setLayout(self.LayoutForTabPageA5)
        #
        for idict in range(len(self.LineEditModelParamsDicts)):
            this_layout = QGridLayout()
            this_layout.setHorizontalSpacing(30)
            this_layout.setVerticalSpacing(5)
            ncolumn = 2
            icount = 0
            for key in self.LineEditModelParamsDicts[idict]:
                if not (key.find('NULL')>=0):
                    self.LineEditModelParamsDicts[idict][key].ParamUpdateSignal.connect(self.onParamUpdateCall)
                    if self.LineEditModelParamsDicts[idict][key].isEnabled():
                        self.DysmalPyParams[key] = self.LineEditModelParamsDicts[idict][key].keyvalue()
                    icount, irow, icol, rowSpan, colSpan = self.LineEditModelParamsDicts[idict][key].getPositionInQGridLayout(icount, ncolumn)
                    this_layout.addWidget(self.LineEditModelParamsDicts[idict][key], irow, icol, rowSpan, colSpan)
                else:
                    icount += 1
            this_tabpage_layout = QVBoxLayout()
            this_tabpage_layout.setContentsMargins(0, 0, 0, 0)
            this_tabpage_layout.setSpacing(0)
            this_tabpage_layout.addItem(this_layout)
            this_tabpage_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
            getattr(self, 'TabPageB%d'%(idict+1)).setLayout(this_tabpage_layout)
        #
        #self.LayoutForLineEditModelParams2 = QGridLayout()
        #self.LayoutForLineEditModelParams2.setHorizontalSpacing(30)
        #self.LayoutForLineEditModelParams2.setVerticalSpacing(5)
        #ncolumn = 2
        #icount = 0
        #for key in self.LineEditModelParamsDict2:
        #    if not (key.find('NULL')>=0):
        #        self.LineEditModelParamsDict2[key].ParamUpdateSignal.connect(self.onParamUpdateCall)
        #        if self.LineEditModelParamsDict2[key].isEnabled():
        #            self.DysmalPyParams[key] = self.LineEditModelParamsDict2[key].keyvalue()
        #        icount, irow, icol, rowSpan, colSpan = self.LineEditModelParamsDict2[key].getPositionInQGridLayout(icount, ncolumn)
        #        self.LayoutForLineEditModelParams2.addWidget(self.LineEditModelParamsDict2[key], irow, icol, rowSpan, colSpan)
        #    else:
        #        icount += 1
        #self.LayoutForTabPageB2 = QVBoxLayout()
        #self.LayoutForTabPageB2.setContentsMargins(0, 0, 0, 0)
        #self.LayoutForTabPageB2.setSpacing(0)
        #self.LayoutForTabPageB2.addItem(self.LayoutForLineEditModelParams2)
        #self.LayoutForTabPageB2.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        #self.TabPageB2.setLayout(self.LayoutForTabPageB2)
        #
        #self.LayoutForLineEditModelParams3 = QGridLayout()
        #self.LayoutForLineEditModelParams3.setHorizontalSpacing(30)
        #self.LayoutForLineEditModelParams3.setVerticalSpacing(5)
        #ncolumn = 2
        #icount = 0
        #for key in self.LineEditModelParamsDict3:
        #    if not (key.find('NULL')>=0):
        #        self.LineEditModelParamsDict3[key].ParamUpdateSignal.connect(self.onParamUpdateCall)
        #        if self.LineEditModelParamsDict3[key].isEnabled():
        #            self.DysmalPyParams[key] = self.LineEditModelParamsDict3[key].keyvalue()
        #        icount, irow, icol, rowSpan, colSpan = self.LineEditModelParamsDict3[key].getPositionInQGridLayout(icount, ncolumn)
        #        self.LayoutForLineEditModelParams3.addWidget(self.LineEditModelParamsDict3[key], irow, icol, rowSpan, colSpan)
        #    else:
        #        icount += 1
        #self.LayoutForTabPageB3 = QVBoxLayout()
        #self.LayoutForTabPageB3.setContentsMargins(0, 0, 0, 0)
        #self.LayoutForTabPageB3.setSpacing(0)
        #self.LayoutForTabPageB3.addItem(self.LayoutForLineEditModelParams3)
        #self.LayoutForTabPageB3.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        #self.TabPageB3.setLayout(self.LayoutForTabPageB3)
        #
        #self.LayoutForLineEditModelParams4 = QGridLayout()
        #self.LayoutForLineEditModelParams4.setHorizontalSpacing(30)
        #self.LayoutForLineEditModelParams4.setVerticalSpacing(5)
        #ncolumn = 2
        #icount = 0
        #for key in self.LineEditModelParamsDict4:
        #    if not (key.find('NULL')>=0):
        #        self.LineEditModelParamsDict4[key].ParamUpdateSignal.connect(self.onParamUpdateCall)
        #        if self.LineEditModelParamsDict4[key].isEnabled():
        #            self.DysmalPyParams[key] = self.LineEditModelParamsDict4[key].keyvalue()
        #        icount, irow, icol, rowSpan, colSpan = self.LineEditModelParamsDict4[key].getPositionInQGridLayout(icount, ncolumn)
        #        self.LayoutForLineEditModelParams4.addWidget(self.LineEditModelParamsDict4[key], irow, icol, rowSpan, colSpan)
        #    else:
        #        icount += 1
        #self.LayoutForTabPageB4 = QVBoxLayout()
        #self.LayoutForTabPageB4.setContentsMargins(0, 0, 0, 0)
        #self.LayoutForTabPageB4.setSpacing(0)
        #self.LayoutForTabPageB4.addItem(self.LayoutForLineEditModelParams4)
        #self.LayoutForTabPageB4.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        #self.TabPageB4.setLayout(self.LayoutForTabPageB4)
        #
        self.LayoutForTabWidgetHolderA = QVBoxLayout()
        self.LayoutForTabWidgetHolderA.addWidget(self.LabelForTabWidgetA)
        self.LayoutForTabWidgetHolderA.addWidget(self.TabWidgetA)
        self.TabWidgetHolderA.setLayout(self.LayoutForTabWidgetHolderA)
        self.LayoutForTabWidgetHolderB = QVBoxLayout()
        self.LayoutForTabWidgetHolderB.addWidget(self.LabelForTabWidgetB)
        self.LayoutForTabWidgetHolderB.addWidget(self.TabWidgetB)
        self.TabWidgetHolderB.setLayout(self.LayoutForTabWidgetHolderB)
        self.SplitterForTabWidgets = QSplitter(Qt.Vertical)
        self.SplitterForTabWidgets.addWidget(self.TabWidgetHolderA)
        self.SplitterForTabWidgets.addWidget(self.TabWidgetHolderB)
        self.SplitterForTabWidgets.setStyleSheet("""
            QSplitter::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #eee, stop:1 #ccc);
                border: 1px solid #aaa;
                width: 20px;
                margin-top: 2px;
                margin-bottom: 2px;
                border-radius: 4px;
            }""")
        self.LayoutForPanelLeft = QVBoxLayout()
        self.LayoutForPanelLeft.addWidget(self.SplitterForTabWidgets)
        self.LayoutForPanelLeft.setContentsMargins(0, 0, 0, 0)
        self.PanelLeft.setLayout(self.LayoutForPanelLeft)
        #
        # Panel Middle widgets
        self.LabelForControlButtons = QLabel(self.tr("Control"))
        self.LabelComponents = QLabel(self.tr("Components"))
        #
        self.CheckBoxModelParamsDict = OrderedDict()
        #self.CheckBoxModelParamsDict['include_halo'] = QWidgetForParamInput(\
        #             keyname=self.tr("include_halo"),
        #             keycomment=self.tr("Include the halo as a component in fitting?"),
        #             datatype=bool,
        #             checkbox=True,
        #             default='True')
        self.CheckBoxModelParamsDict['adiabatic_contract'] = QWidgetForParamInput(\
                     keyname=self.tr("adiabatic_contract"),
                     keycomment=self.tr("Apply adiabatic contraction?"),
                     datatype=bool,
                     checkbox=True)
        self.CheckBoxModelParamsDict['pressure_support'] = QWidgetForParamInput(\
                     keyname=self.tr("pressure_support"),
                     keycomment=self.tr("Apply assymmetric drift correction?"),
                     datatype=bool,
                     checkbox=True,
                     default='True')
        self.CheckBoxModelParamsDict['noord_flat'] = QWidgetForParamInput(\
                     keyname=self.tr("noord_flat"),
                     keycomment=self.tr("Apply Noordermeer 2008 flattenning for the velocity field of Sersic profile? (Need \"$SERSIC_PROFILE_MASS_VC_DATADIR/mass_VC_profile_sersic_n*.fits\")"),
                     datatype=bool,
                     checkbox=True,
                     default='True')
        self.CheckBoxModelParamsDict['zheight_tied'] = QWidgetForParamInput(\
                     keyname=self.tr('zheight_tied'),
                     keycomment=self.tr("Tie the zheight to the effective radius of the disk?"),
                     datatype=bool,
                     checkbox=True,
                     default='True')
        self.CheckBoxModelParamsDict['zheight_tied'].CheckBoxWidget.stateChanged.disconnect()
        self.CheckBoxModelParamsDict['zheight_tied'].CheckBoxWidget.stateChanged.connect(self.onZheightTiedCheckStateChangedCall)
        self.CheckBoxModelParamsDict['mvirial_tied'] = QWidgetForParamInput(\
                     keyname=self.tr('mvirial_tied'),
                     keycomment=self.tr("For NFW, mvirial_tied=True determines Mvirial from fDM (at r_eff_disk)."),
                     datatype=bool,
                     checkbox=True,
                     default='False')
        self.CheckBoxModelParamsDict['mvirial_tied'].CheckBoxWidget.stateChanged.disconnect()
        self.CheckBoxModelParamsDict['mvirial_tied'].CheckBoxWidget.stateChanged.connect(self.onMvirialTiedCheckStateChangedCall)
        self.CheckBoxModelParamsDict['fdm_tied'] = QWidgetForParamInput(\
                     keyname=self.tr('fdm_tied'),
                     keycomment=self.tr("For NFW, fdm_tied=True determines fDM from Mvirial (+baryons)."),
                     datatype=bool,
                     checkbox=True,
                     default='True')
        self.CheckBoxModelParamsDict['fdm_tied'].CheckBoxWidget.stateChanged.disconnect()
        self.CheckBoxModelParamsDict['fdm_tied'].CheckBoxWidget.stateChanged.connect(self.onFdmTiedCheckStateChangedCall)
        self.CheckBoxModelParamsDict['zcalc_truncate'] = QWidgetForParamInput(\
                     keyname=self.tr('zcalc_truncate'),
                     keycomment=self.tr("If True, the cube is only filled with flux to within +- 2 * scale length thickness above and below the galaxy midplane"),
                     datatype=bool,
                     checkbox=True,
                     default='True')
        self.CheckBoxModelParamsDict['zcalc_with_c'] = QWidgetForParamInput( \
                    keyname=self.tr('zcalc_with_c'),
                    keycomment=self.tr("If True, the cube is populated with c++ code."),
                    datatype=bool,
                    checkbox=True,
                    default='False')
        self.CheckBoxModelParamsDict['partial_weight'] = QWidgetForParamInput(\
                     keyname=self.tr('partial_weight'),
                     keycomment=self.tr("If True, do partial weight when dealing with aperture inner pixels."),
                     datatype=bool,
                     checkbox=True,
                     default='True')
        #
        self.CheckBoxModelParamsDict['__gauss_extract__'] = QWidgetForParamInput(\
                     keyname=self.tr('__gauss_extract__'),
                     keycomment=self.tr("Using Gaussian line profile fitting instead of moment to extract the velocity field. Inverse of moment_calc."),
                     datatype=bool,
                     checkbox=True,
                     default='True')
        self.CheckBoxModelParamsDict['__gauss_extract__'].CheckBoxWidget.stateChanged.disconnect()
        self.CheckBoxModelParamsDict['__gauss_extract__'].CheckBoxWidget.stateChanged.connect(self.onGaussExtractCheckStateChangedCall)
        #
        self.CheckBoxModelParamsDict['gauss_extract_with_c'] = QWidgetForParamInput(\
                     keyname=self.tr('gauss_extract_with_c'),
                     keycomment=self.tr("Using C++-based Gaussian line profile fitting to speed up. "),
                     datatype=bool,
                     checkbox=True,
                     default='True')
        #self.CheckBoxModelParamsDict['gauss_extract_with_c'].CheckBoxWidget.stateChanged.disconnect() # do not disconnect here, so that onParamUpdate will also be called
        self.CheckBoxModelParamsDict['gauss_extract_with_c'].CheckBoxWidget.stateChanged.connect(self.onGaussExtractCheckStateChangedCall)
        self.LineEditModelParamsDictForFitting['moment_calc'].ComboBoxWidget.currentTextChanged.connect(self.onMomentCalcLineEditTextChangedCall)
        #
        self.CheckBoxModelParamsDict['__oversampling__'] = QWidgetForParamInput(\
                     keyname=self.tr('__oversampling__'),
                     keycomment=self.tr("Oversampling the model cube. See also the oversample input box."),
                     datatype=bool,
                     checkbox=True,
                     default=str(self.LineEditModelParamsDictForFitting['oversample'].keyvalue()>1))
        self.CheckBoxModelParamsDict['__oversampling__'].CheckBoxWidget.stateChanged.disconnect()
        self.CheckBoxModelParamsDict['__oversampling__'].CheckBoxWidget.stateChanged.connect(self.onOverSamplingCheckStateChangedCall)
        #
        self.CheckBoxModelParamsDict['__overwriting__'] = QWidgetForParamInput(\
                     keyname=self.tr('__overwriting__'),
                     keycomment=self.tr("Overwriting the fitting. See also the overwrite check box."),
                     datatype=bool,
                     checkbox=True,
                     default=self.LineEditModelParamsDictForFitting['overwrite'].keyvalue())
        self.CheckBoxModelParamsDict['__overwriting__'].CheckBoxWidget.stateChanged.disconnect()
        self.CheckBoxModelParamsDict['__overwriting__'].CheckBoxWidget.stateChanged.connect(self.onOverWritingCheckStateChangedCall)
        self.LineEditModelParamsDictForFitting['overwrite'].ComboBoxWidget.currentTextChanged.connect(self.onOverWritingLineEditTextChangedCall)
        #
        #
        self.ButtonInitRandomParams = QPushButton(self.tr("Init Random Params"))
        self.ButtonInitRandomParams.setMinimumWidth(160)
        self.ButtonInitRandomParams.clicked.connect(self.onInitRandomParamsCall)
        self.ButtonOpenParamsFile = QPushButton(self.tr("Open Params File"))
        self.ButtonOpenParamsFile.setMinimumWidth(160)
        self.ButtonOpenParamsFile.clicked.connect(self.onOpenParamFileCall)
        self.ButtonFitData = QPushButton(self.tr("Fit Data"))
        self.ButtonFitData.setMinimumWidth(160)
        self.ButtonFitData.clicked.connect(self.onFitDataCall)
        self.ButtonLoadFittingResult = QPushButton(self.tr("Load Fitting Result"))
        self.ButtonLoadFittingResult.setMinimumWidth(160)
        self.ButtonLoadFittingResult.clicked.connect(self.onLoadFittingResultCall)
        self.ButtonGenerateModelCube = QPushButton(self.tr("Generate Model Cube"))
        self.ButtonGenerateModelCube.setMinimumWidth(160)
        self.ButtonGenerateModelCube.clicked.connect(self.onGenerateModelCubeCall)
        self.ButtonGenerateMomentMaps = QPushButton(self.tr("Generate Mom. Maps"))
        self.ButtonGenerateMomentMaps.setMinimumWidth(160)
        self.ButtonGenerateMomentMaps.clicked.connect(self.onGenerateMomentMapsCall)
        self.ButtonGenerateRotationCurves = QPushButton(self.tr("Generate Rot. Curves"))
        self.ButtonGenerateRotationCurves.setMinimumWidth(160)
        self.ButtonGenerateRotationCurves.clicked.connect(self.onGenerateRotationCurvesCall)
        self.ButtonHideSlit = QPushButton(self.tr("Hide Slit"))
        self.ButtonHideSlit.setMinimumWidth(160)
        self.ButtonHideSlit.clicked.connect(self.onHideSlitCall)
        self.ButtonSaveParams = QPushButton(self.tr("Save Params"))
        self.ButtonSaveParams.setMinimumWidth(160)
        self.ButtonSaveParams.clicked.connect(self.onSaveParamFileCall)
        self.ButtonSaveModelFiles = QPushButton(self.tr("Save Model Files"))
        self.ButtonSaveModelFiles.setToolTip(self.tr("Save model cube, flux map, velocity map and dispersion map as FITS files."))
        self.ButtonSaveModelFiles.setMinimumWidth(160)
        self.ButtonSaveModelFiles.clicked.connect(self.onSaveModelFilesCall)
        self.ButtonExit = QPushButton(self.tr("Exit"))
        self.ButtonExit.setMinimumWidth(160)
        self.ButtonExit.clicked.connect(self.onExitCall)
        #
        # Panel Middle layout
        self.LayoutForCheckBoxModelParams = QGridLayout()
        self.LayoutForCheckBoxModelParams.setContentsMargins(0, 0, 0, 0)
        self.LayoutForCheckBoxModelParams.setHorizontalSpacing(30)
        self.LayoutForCheckBoxModelParams.setVerticalSpacing(5)
        self.LayoutForCheckBoxModelParams.setSizeConstraint(QLayout.SetMinimumSize)
        ncolumn = 1
        icount = 0
        for key in self.CheckBoxModelParamsDict:
            if not (key.find('NULL')>=0):
                self.CheckBoxModelParamsDict[key].setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
                self.CheckBoxModelParamsDict[key].ParamUpdateSignal.connect(self.onParamUpdateCall)
                if self.CheckBoxModelParamsDict[key].isEnabled():
                    self.DysmalPyParams[key] = self.CheckBoxModelParamsDict[key].keyvalue()
                    logger.debug('self.DysmalPyParams[\'%s\'] = self.CheckBoxModelParamsDict[\'%s\'].keyvalue() = %s'%(key, key, self.CheckBoxModelParamsDict[key].keyvalue()))
                icount, irow, icol, rowSpan, colSpan = self.CheckBoxModelParamsDict[key].getPositionInQGridLayout(icount, ncolumn)
                self.LayoutForCheckBoxModelParams.addWidget(self.CheckBoxModelParamsDict[key], irow, icol, rowSpan, colSpan)
            else:
                icount += 1
        #self.LayoutForCheckBoxModelParamsHolder = QHBoxLayout()
        #self.LayoutForCheckBoxModelParamsHolder.addItem(self.LayoutForCheckBoxModelParams)
        #self.LayoutForCheckBoxModelParamsHolder.addItem(QSpacerItem(2, 40, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.LayoutForPanelMiddle = QVBoxLayout()
        self.LayoutForPanelMiddle.setAlignment(Qt.AlignCenter)
        self.LayoutForPanelMiddle.setContentsMargins(0, 0, 0, 0)
        self.LayoutForPanelMiddle.addWidget(self.LabelForControlButtons)
        self.LayoutForPanelMiddle.addItem(self.LayoutForCheckBoxModelParams) # self.LayoutForCheckBoxModelParamsHolder
        self.LayoutForPanelMiddle.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.LayoutForPanelMiddle.addWidget(self.ButtonInitRandomParams)
        self.LayoutForPanelMiddle.addWidget(self.ButtonOpenParamsFile)
        self.LayoutForPanelMiddle.addWidget(self.ButtonFitData)
        self.LayoutForPanelMiddle.addWidget(self.ButtonLoadFittingResult)
        self.LayoutForPanelMiddle.addWidget(self.ButtonGenerateModelCube)
        self.LayoutForPanelMiddle.addWidget(self.ButtonGenerateMomentMaps)
        self.LayoutForPanelMiddle.addWidget(self.ButtonGenerateRotationCurves)
        self.LayoutForPanelMiddle.addWidget(self.ButtonHideSlit)
        self.LayoutForPanelMiddle.addWidget(self.ButtonSaveParams)
        self.LayoutForPanelMiddle.addWidget(self.ButtonSaveModelFiles)
        self.LayoutForPanelMiddle.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.LayoutForPanelMiddle.addWidget(self.ButtonExit)
        self.PanelMiddle.setLayout(self.LayoutForPanelMiddle)
        self.PanelMiddle.setStyleSheet("""
            QPushButton {
                max-width: 100px;
            }
            QCheckBox {
                max-width: 100px;
            }
            """)
        #
        # Panel Right widgets
        self.LabelForSpecViewerA = QLabel(self.tr("Rotation Curves"))
        self.LabelForImageViewerA = QLabel(self.tr("Data Images"))
        self.LabelForImageViewerB = QLabel(self.tr("Model Images"))
        self.LabelForImageViewerC = QLabel(self.tr("Residual Images"))
        self.ImageViewerA1 = QFitsImageWidget(title='Flux', name='Data Flux Map', with_colorbar=True)
        self.ImageViewerA2 = QFitsImageWidget(title='Vel', name='Data Vel Map', with_colorbar=True)
        self.ImageViewerA3 = QFitsImageWidget(title='Disp', name='Data Disp Map', with_colorbar=True)
        self.ImageViewerB1 = QFitsImageWidget(title='Flux', name='Model Flux Map', with_colorbar=True)
        self.ImageViewerB2 = QFitsImageWidget(title='Vel', name='Model Vel Map', with_colorbar=True)
        self.ImageViewerB3 = QFitsImageWidget(title='Disp', name='Model Disp Map', with_colorbar=True)
        self.ImageViewerC1 = QFitsImageWidget(title='Flux', name='Residual Flux Map', with_colorbar=True)
        self.ImageViewerC2 = QFitsImageWidget(title='Vel', name='Residual Vel Map', with_colorbar=True)
        self.ImageViewerC3 = QFitsImageWidget(title='Disp', name='Residual Disp Map', with_colorbar=True)
        self.ImageViewerA1.PixelSelectedSignal.connect(self.selectPixelInImageViewers)
        self.ImageViewerA2.PixelSelectedSignal.connect(self.selectPixelInImageViewers)
        self.ImageViewerA3.PixelSelectedSignal.connect(self.selectPixelInImageViewers)
        self.ImageViewerB1.PixelSelectedSignal.connect(self.selectPixelInImageViewers)
        self.ImageViewerB2.PixelSelectedSignal.connect(self.selectPixelInImageViewers)
        self.ImageViewerB3.PixelSelectedSignal.connect(self.selectPixelInImageViewers)
        self.ImageViewerC1.PixelSelectedSignal.connect(self.selectPixelInImageViewers)
        self.ImageViewerC2.PixelSelectedSignal.connect(self.selectPixelInImageViewers)
        self.ImageViewerC3.PixelSelectedSignal.connect(self.selectPixelInImageViewers)
        self.SpecViewerA1 = QSpectrumWidget(title='Flux', name='1D Flux')
        self.SpecViewerA2 = QSpectrumWidget(title='Vel', name='1D Vel')
        self.SpecViewerA3 = QSpectrumWidget(title='Disp', name='1D Disp')
        self.SpecViewerA1.ChannelSelectedSignal.connect(self.selectChannelInSpecViewers)
        self.SpecViewerA2.ChannelSelectedSignal.connect(self.selectChannelInSpecViewers)
        self.SpecViewerA3.ChannelSelectedSignal.connect(self.selectChannelInSpecViewers)
        #
        # Panel Right layout
        self.LayoutForSpecViewerA = QHBoxLayout()
        self.LayoutForSpecViewerA.setContentsMargins(0, 0, 0, 0)
        self.LayoutForSpecViewerA.addWidget(self.SpecViewerA1)
        self.LayoutForSpecViewerA.addWidget(self.SpecViewerA2)
        self.LayoutForSpecViewerA.addWidget(self.SpecViewerA3)
        self.LayoutForImageViewerA = QHBoxLayout()
        self.LayoutForImageViewerA.setContentsMargins(0, 0, 0, 0)
        self.LayoutForImageViewerA.addWidget(self.ImageViewerA1)
        self.LayoutForImageViewerA.addWidget(self.ImageViewerA2)
        self.LayoutForImageViewerA.addWidget(self.ImageViewerA3)
        self.LayoutForImageViewerB = QHBoxLayout()
        self.LayoutForImageViewerB.setContentsMargins(0, 0, 0, 0)
        self.LayoutForImageViewerB.addWidget(self.ImageViewerB1)
        self.LayoutForImageViewerB.addWidget(self.ImageViewerB2)
        self.LayoutForImageViewerB.addWidget(self.ImageViewerB3)
        self.LayoutForImageViewerC = QHBoxLayout()
        self.LayoutForImageViewerC.setContentsMargins(0, 0, 0, 0)
        self.LayoutForImageViewerC.addWidget(self.ImageViewerC1)
        self.LayoutForImageViewerC.addWidget(self.ImageViewerC2)
        self.LayoutForImageViewerC.addWidget(self.ImageViewerC3)
        self.LayoutForSpecViewerWithLabelA = QVBoxLayout()
        self.LayoutForImageViewerWithLabelA = QVBoxLayout()
        self.LayoutForImageViewerWithLabelB = QVBoxLayout()
        self.LayoutForImageViewerWithLabelC = QVBoxLayout()
        self.LayoutForSpecViewerWithLabelA.setContentsMargins(0, 0, 0, 0)
        self.LayoutForImageViewerWithLabelA.setContentsMargins(0, 0, 0, 0)
        self.LayoutForImageViewerWithLabelB.setContentsMargins(0, 0, 0, 0)
        self.LayoutForImageViewerWithLabelC.setContentsMargins(0, 0, 0, 0)
        self.LayoutForSpecViewerWithLabelA.addWidget(self.LabelForSpecViewerA)
        self.LayoutForSpecViewerWithLabelA.addItem(self.LayoutForSpecViewerA)
        self.LayoutForImageViewerWithLabelA.addWidget(self.LabelForImageViewerA)
        self.LayoutForImageViewerWithLabelA.addItem(self.LayoutForImageViewerA)
        self.LayoutForImageViewerWithLabelB.addWidget(self.LabelForImageViewerB)
        self.LayoutForImageViewerWithLabelB.addItem(self.LayoutForImageViewerB)
        self.LayoutForImageViewerWithLabelC.addWidget(self.LabelForImageViewerC)
        self.LayoutForImageViewerWithLabelC.addItem(self.LayoutForImageViewerC)
        self.LayoutForPanelRight = QVBoxLayout()
        self.LayoutForPanelRight.setContentsMargins(0, 0, 0, 0)
        self.LayoutForPanelRight.addItem(self.LayoutForSpecViewerWithLabelA)
        self.LayoutForPanelRight.addItem(self.LayoutForImageViewerWithLabelA)
        self.LayoutForPanelRight.addItem(self.LayoutForImageViewerWithLabelB)
        self.LayoutForPanelRight.addItem(self.LayoutForImageViewerWithLabelC)
        self.LayoutForPanelRight.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.PanelRight.setLayout(self.LayoutForPanelRight)
        #
        #self.SplitterA = QSplitter(Qt.Horizontal)
        #self.SplitterB = QSplitter(Qt.Horizontal)
        #self.SplitterA.addWidget(self.PanelImageA)
        #self.SplitterB.addWidget(self.PanelImageB)
        #self.SplitterA.addWidget(self.PanelSpecA)
        #self.SplitterB.addWidget(self.PanelSpecB)
        #self.SplitterA.setStretchFactor(0, 1)
        #self.SplitterB.setStretchFactor(0, 1)
        #self.SplitterA.setStretchFactor(1, 3)
        #self.SplitterB.setStretchFactor(1, 3)
        #self.SplitterA.splitterMoved.connect(self.moveSplitterAB)
        #self.SplitterB.splitterMoved.connect(self.moveSplitterAB)
        #
        self.SplitterForCentralWidget = QSplitter(Qt.Horizontal)
        self.SplitterForCentralWidget.addWidget(self.PanelLeft)
        self.SplitterForCentralWidget.addWidget(self.PanelMiddle)
        self.SplitterForCentralWidget.addWidget(self.PanelRight)
        self.SplitterForCentralWidget.setStretchFactor(0, 6)
        self.SplitterForCentralWidget.setStretchFactor(1, 0)
        self.SplitterForCentralWidget.setStretchFactor(2, 6)
        self.SplitterForCentralWidget.setSizes([9, 1, 9])
        self.SplitterForCentralWidget.setStyleSheet("""
            QSplitter::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #eee, stop:1 #ccc);
                border: 1px solid #aaa;
                width: 20px;
                margin-top: 2px;
                margin-bottom: 2px;
                border-radius: 4px;
            }""")
        self.PanelLeft.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.PanelMiddle.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.PanelRight.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        #
        # set main layout
        self.LayoutForCentralWidget = QHBoxLayout()
        self.LayoutForCentralWidget.addWidget(self.SplitterForCentralWidget)
        self.LayoutForCentralWidget.setContentsMargins(0, 0, 0, 0)
        self.LayoutForCentralWidget.setSpacing(0)
        self.CentralWidget.setLayout(self.LayoutForCentralWidget)
        self.setCentralWidget(self.CentralWidget)
        #
        #self.adjustSize()
        #
        # set status bar
        self.logMessage(self.tr('Initializing...'))
        #
        # set menu actions
        exitAct = QAction(self.tr('&Exit'), self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip(self.tr('Exit application'))
        exitAct.setMenuRole(QAction.NoRole) # see https://stackoverflow.com/questions/39574105/missing-menubar-in-pyqt5
        exitAct.triggered.connect(self.onExitCall)
        openParamFileAct = QAction(self.tr('&Open Param File'), self)
        openParamFileAct.setShortcut('Ctrl+O')
        openParamFileAct.setStatusTip(self.tr('Open a *.params file'))
        openParamFileAct.triggered.connect(self.onOpenParamFileCall)
        #
        # set menu
        menubar = QMenuBar(self)
        menubar.setNativeMenuBar(False)
        fileMenu = QMenu(self.tr('&File'), menubar)
        fileMenu.addAction(openParamFileAct)
        fileMenu.addAction(exitAct)
        menubar.addMenu(fileMenu)
        self.setMenuBar(menubar)
        #
        # set fitting worker qobject
        self.DysmalPyFittingTower.LogMessageUpdateSignal.connect(self.onLogMessageUpdateCall)
        self.LineEditDataParamsDict['outdir'].ParamUpdateSignal.connect(self.onOutDirParamUpdateCall)
        #
        # set status bar
        self.logMessage(self.tr('Ready'))
        #
        # set main window geometry size to 95% x 85% of the screen size
        if ScreenSize is not None:
            self.setGeometry(int(ScreenSize.width() * (1.0-0.95)/2.0), 
                             int(ScreenSize.height() * (1.0-0.85)/2.0), 
                             int(ScreenSize.width() * 0.95), 
                             int(ScreenSize.height() * 0.85) )
            # make it centered on screen, 75% screen width and 85% screen height.
        #self.setWindowTitle('Icon')
        #self.setWindowIcon(QIcon('web.png'))
        self.setWindowTitle("DysmalPy - 3D Forward Modeling & Kinematic Fitting - MPE IR Group")
        #self.setWindowIcon(getIconDysmalPy())
        # 
        self.SplitterForCentralWidget.setSizes([9, 1, 9])
        #
        self.setButtonsEnabledDisabled()
        #
        #self.ShortCutOpenParamFile = QShortcut(QKeySequence('Alt+O'), self)
        #self.ShortCutOpenParamFile.activated.connect(self.onOpenParamFileCall)
        #
        self.setAcceptDrops(True)
        #
        self.show()

    def moveSplitterAB(self, index, pos):
        self.SplitterA.blockSignals(True)
        self.SplitterB.blockSignals(True)
        self.SplitterA.moveSplitter(index, pos)
        self.SplitterB.moveSplitter(index, pos)
        self.SplitterA.blockSignals(False)
        self.SplitterB.blockSignals(False)

    def closeEvent(self, event):
        if self.DysmalPyFittingTower is not None:
            if self.DysmalPyFittingTower.BaseQueue is not None:
                self.DysmalPyFittingTower.BaseQueue.put( ('signal', 'exit') )
            time.sleep(0.5) #<TODO># check child thread cleared?
            del self.DysmalPyFittingTower
            self.DysmalPyFittingTower = None
        self.logger.debug('Closing... Bye.')
        super(QDysmalPyGUI, self).closeEvent(event)

    @pyqtSlot()
    def onExitCall(self):
        self.logger.debug('Closing on Exit call.')
        self.close()
        # qApp.quit()

    def logMessage(self, message):
        if self.statusBar():
            self.logger.info(message)
            self.statusBar().showMessage(message)
        
    @pyqtSlot(str)
    def onLogMessageUpdateCall(self, message):
        self.logMessage(message)
        
    @pyqtSlot(str, str, type, type)
    def onParamUpdateCall(self, keyname, keyvalue, datatype, listtype):
        if keyname.find('NULL')>=0 or keyname.startswith('__'):
            return
        if self.DysmalPyParams is None:
            self.DysmalPyParams = {} # create an empty DysmalPyParams
        
        if keyvalue == '':
            keyvalue = None
        elif keyvalue == 'None':
            keyvalue = None
        elif keyvalue == 'True':
            keyvalue = True
        elif keyvalue == 'False':
            keyvalue = False
        elif keyvalue == 'inf':
            keyvalue = np.inf
        else:
            if keyname not in self.DysmalPyParams:
                self.DysmalPyParams[keyname] = None
            if listtype is list:
                try:
                    keyvalue = eval(keyvalue)
                    keyvalue = np.array(keyvalue).astype(datatype)
                except:
                    self.logMessage(self.tr('Could not update DysmalPyParams key ')+str(keyname)+
                                    self.tr(' value ')+'"'+str(keyvalue)+'"'+self.tr(' as type ')+str(datatype)+
                                    self.tr('. Current value '+str(self.DysmalPyParams[keyname])))
                    return
            else:
                try:
                    keyvalue = np.array([keyvalue]).astype(datatype)[0]
                except:
                    self.logMessage(self.tr('Could not update DysmalPyParams key ')+str(keyname)+
                                    self.tr(' value ')+'"'+str(keyvalue)+'"'+self.tr(' as type ')+str(datatype)+
                                    self.tr('. Current value '+str(self.DysmalPyParams[keyname])))
                    return
        
        # for data related keys, if keyvalue is set to None, then we delete the key in the dict DysmalPyParams
        if keyname in ['fdata', 'fdata_flux', 'fdata_ferr', 'fdata_vel', 'fdata_verr', 'fdata_disp', 'fdata_derr', 'fdata_mask', 'fdata_cube', 'fdata_err'] and \
           keyname in self.DysmalPyParams and \
           keyvalue is None:
            del self.DysmalPyParams[keyname]
            self.logMessage(self.tr('Deleted DysmalPyParams key ')+str(keyname))
        # for lensing related keys, if any item is changed, then we reset lensing_transformer
        elif keyname.startswith('lensing_') and \
             keyvalue != self.DysmalPyParams[keyname]:
            self.clearLensingTransformer()
            self.logMessage(self.tr('Cleared DysmalPyFittingTower lensing_transformer due to changing key ')+str(keyname))
        # for additional component -- non-circular motion keys, manipulate the 'components_list'
        elif keyname.startswith('include_'):
            addcomponent = keyname.replace('include_', '')
            if keyvalue == True:
                self.logMessage(self.tr('Enabling component ')+str(addcomponent))
                if 'components_list' not in self.DysmalPyParams:
                    self.DysmalPyParams['components_list'] = ['disk+bulge', 'const_disp_prof', 'geometry', 'zheight_gaus']
                    if self.DysmalPyParams['include_halo']:
                        self.DysmalPyParams['components_list'].append('halo')
                if addcomponent not in self.DysmalPyParams['components_list']:
                    self.DysmalPyParams['components_list'].append(addcomponent)
                self.logMessage(self.tr('Updated DysmalPyParams key ')+str('components_list')+
                                self.tr(' value ')+str(self.DysmalPyParams['components_list'])+self.tr(' type ')+str('list'))
            else:
                self.logMessage(self.tr('Disabling component ')+str(addcomponent))
                if 'components_list' in self.DysmalPyParams:
                    if addcomponent in self.DysmalPyParams['components_list']:
                        self.DysmalPyParams['components_list'].remove(addcomponent)
                    self.logMessage(self.tr('Updated DysmalPyParams key ')+str('components_list')+
                                    self.tr(' value ')+str(self.DysmalPyParams['components_list'])+self.tr(' type ')+str('list'))
            #print('self.LineEditModelParamsDictForNonCirc[keyname].AssociatedParams', self.LineEditModelParamsDictForNonCirc[keyname].AssociatedParams)
            if keyname in self.LineEditModelParamsDictForNonCirc:
                associatedparams = self.LineEditModelParamsDictForNonCirc[keyname].AssociatedParams
                if associatedparams is not None:
                    for associatedkeyname in associatedparams:
                        self.LineEditModelParamsDictForNonCirc[associatedkeyname].setEnabled(keyvalue)
                        if keyvalue == True:
                            associatedkeyvalue = self.LineEditModelParamsDictForNonCirc[associatedkeyname].keyvalue()
                            associateddatatype = self.LineEditModelParamsDictForNonCirc[associatedkeyname].dataType()
                            self.DysmalPyParams[associatedkeyname] = associatedkeyvalue
                            self.logMessage(self.tr('Updated DysmalPyParams key ')+str(associatedkeyname)+
                                            self.tr(' value ')+str(associatedkeyvalue)+self.tr(' type ')+str(associateddatatype))
                        else:
                            if associatedkeyname in self.DysmalPyParams:
                                del self.DysmalPyParams[associatedkeyname]
                                self.logMessage(self.tr('Deleted DysmalPyParams key ')+str(associatedkeyname))
        else:
            
            # for data dir out dir ensure ending with slash
            if keyname in ['datadir', 'outdir']:
                if keyvalue is not None:
                    keyvalue = ensure_path_trailing_slash(keyvalue)
            
            self.DysmalPyParams[keyname] = keyvalue
            self.logMessage(self.tr('Updated DysmalPyParams key ')+str(keyname)+
                            self.tr(' value ')+str(keyvalue)+self.tr(' type ')+str(datatype))
        # 
        # for oversample key, also update self.CheckBoxModelParamsDict['__oversampling__']
        if keyname == 'oversample':
            self.CheckBoxModelParamsDict['__oversampling__'].setChecked(keyvalue>1, blocksignal=True)
    
    @pyqtSlot(int)
    def onGaussExtractCheckStateChangedCall(self, state):
        self.logger.debug('QWidgetForParamInput::onGaussExtractCheckStateChangedCall()')
        #for this_dict in self.LineEditModelParamsDicts:
        #    for this_key in this_dict:
        #        if this_key == 'moment_calc':
        #            this_dict[this_key].setText(str(np.invert(state>0)))
        state_str = str(self.CheckBoxModelParamsDict['__gauss_extract__'].keyvalue() == False and \
                        self.CheckBoxModelParamsDict['gauss_extract_with_c'].keyvalue() == False)
        is_changed = False
        for this_dict in self.LineEditModelParamsDicts:
            for this_key in this_dict:
                if this_key == 'moment_calc':
                    this_dict[this_key].setText(state_str)
                    is_changed = True
                if is_changed:
                    break
            if is_changed:
                break
    
    @pyqtSlot(str)
    def onMomentCalcLineEditTextChangedCall(self, text):
        self.logger.debug('QWidgetForParamInput::onMomentCalcLineEditTextChangedCall()')
        state_str = (text != 'True') # if moment_calc is true, then we set checkbox to false
        for this_key in self.CheckBoxModelParamsDict:
            if this_key == '__gauss_extract__':
                self.CheckBoxModelParamsDict[this_key].setText(state_str, blocksignal=True)
            if this_key == 'gauss_extract_with_c':
                self.CheckBoxModelParamsDict[this_key].setText(state_str, blocksignal=True)
    
    @pyqtSlot(int)
    def onOverSamplingCheckStateChangedCall(self, state):
        self.logger.debug('QWidgetForParamInput::onOverSamplingCheckStateChangedCall()')
        is_changed = False
        for this_dict in self.LineEditModelParamsDicts:
            for this_key in this_dict:
                if this_key == 'oversample':
                    if state>0:
                        this_dict[this_key].setEnabled(True)
                        this_dict[this_key].setText(str(this_dict[this_key].ParamValue), blocksignal=True)
                    else:
                        this_dict[this_key].ParamValue = this_dict[this_key].keyvalue()
                        this_dict[this_key].setText('1', blocksignal=True)
                        this_dict[this_key].setEnabled(False)
                    is_changed = True
                if is_changed:
                    break
            if is_changed:
                break
    
    @pyqtSlot(int)
    def onOverWritingCheckStateChangedCall(self, state):
        self.logger.debug('QWidgetForParamInput::onOverWritingCheckStateChangedCall()')
        is_changed = False
        for this_dict in self.LineEditModelParamsDicts:
            for this_key in this_dict:
                if this_key == 'overwrite':
                    if state>0:
                        this_dict[this_key].setText('True', blocksignal=True)
                        self.logger.debug('QWidgetForParamInput::onOverWritingCheckStateChangedCall() setting overwrite to True')
                    else:
                        this_dict[this_key].setText('False', blocksignal=True)
                        self.logger.debug('QWidgetForParamInput::onOverWritingCheckStateChangedCall() setting overwrite to False')
                    is_changed = True
                if is_changed:
                    break
            if is_changed:
                break
    
    @pyqtSlot(int)
    def onZheightTiedCheckStateChangedCall(self, state):
        self.logger.debug('QWidgetForParamInput::onZheightTiedCheckStateChangedCall()')
        #for this_dict in self.LineEditModelParamsDicts:
        #    for this_key in this_dict:
        #        if this_key == 'moment_calc':
        #            this_dict[this_key].setText(str(np.invert(state>0)))
        state_tied = (self.CheckBoxModelParamsDict['zheight_tied'].keyvalue() == True)
        is_changed = False
        for this_dict in self.LineEditModelParamsDicts:
            for this_key in this_dict:
                if this_key == 'sigmaz_fixed':
                    if state_tied and this_dict['sigmaz_fixed'].text() == 'True':
                        this_dict['sigmaz_fixed'].setText('False', blocksignal=True) # if tied, it can not be fixed
                        self.logger.debug('QWidgetForParamInput::onZheightTiedCheckStateChangedCall() setting sigmaz_fixed to False')
                    is_changed = True
                if is_changed:
                    break
            if is_changed:
                break
    
    @pyqtSlot(int)
    def onMvirialTiedCheckStateChangedCall(self, state):
        self.logger.debug('QWidgetForParamInput::onMvirialTiedCheckStateChangedCall()')
        #for this_dict in self.LineEditModelParamsDicts:
        #    for this_key in this_dict:
        #        if this_key == 'moment_calc':
        #            this_dict[this_key].setText(str(np.invert(state>0)))
        state_tied = (self.CheckBoxModelParamsDict['mvirial_tied'].keyvalue() == True)
        is_changed = False
        for this_dict in self.LineEditModelParamsDicts:
            for this_key in this_dict:
                if this_key == 'mvirial_fixed':
                    if state_tied and this_dict['mvirial_fixed'].text() == 'True':
                        this_dict['mvirial_fixed'].setText('False', blocksignal=True) # if tied, it can not be fixed
                        self.logger.debug('QWidgetForParamInput::onMvirialTiedCheckStateChangedCall() setting mvirial_fixed to False')
                    is_changed = True
                if is_changed:
                    break
            if is_changed:
                break
    
    @pyqtSlot(int)
    def onFdmTiedCheckStateChangedCall(self, state):
        self.logger.debug('QWidgetForParamInput::onFdmTiedCheckStateChangedCall()')
        #for this_dict in self.LineEditModelParamsDicts:
        #    for this_key in this_dict:
        #        if this_key == 'moment_calc':
        #            this_dict[this_key].setText(str(np.invert(state>0)))
        state_tied = (self.CheckBoxModelParamsDict['fdm_tied'].keyvalue() == True)
        is_changed = False
        for this_dict in self.LineEditModelParamsDicts:
            for this_key in this_dict:
                if this_key == 'fdm_fixed':
                    if state_tied and this_dict['fdm_fixed'].text() == 'True':
                        this_dict['fdm_fixed'].setText('False', blocksignal=True) # if tied, it can not be fixed
                        self.logger.debug('QWidgetForParamInput::onFdmTiedCheckStateChangedCall() setting fdm_fixed to False')
                    is_changed = True
                if is_changed:
                    break
            if is_changed:
                break
    
    @pyqtSlot(str)
    def onOverWritingLineEditTextChangedCall(self, text):
        self.logger.debug('QWidgetForParamInput::onOverWritingLineEditTextChangedCall()')
        for this_key in self.CheckBoxModelParamsDict:
            if this_key == 'overwrite':
                self.CheckBoxModelParamsDict[this_key].setText(text, blocksignal=True)
                break
    
    @pyqtSlot(str, str, type, type)
    def onOutDirParamUpdateCall(self, keyname, keyvalue, datatype, listtype):
        self.logger.debug('QWidgetForParamInput::onOutDirParamUpdateCall()')
        #if keyname == 'outdir':
        #    if keyvalue is not None and keyvalue != '':
        #        self.logger.debug('QWidgetForParamInput::onOutDirParamUpdateCall() self.DysmalPyFittingTower.setDirectory("{}")'.format(keyvalue))
        #        self.DysmalPyFittingTower.setDirectory(keyvalue)
    
    @pyqtSlot()
    def onOpenParamFileCall(self):
        self.logMessage(self.tr('Opening Dysmal param file...'))
        filepath = None
        filepath, selectedfilter = QFileDialog.getOpenFileName(self, self.tr('Open file'), self.DefaultDirectory, self.tr("Dysmal params file (*.params *.*)"))
        if filepath is None or filepath == '':
            self.logMessage(self.tr('No file selected.'))
            self.deselectParamFile()
        else:
            self.logMessage(self.tr('Selected Dysmal param file: '+str(filepath)))
            self.selectParamFile(filepath)

    def selectParamFile(self, filepath):
        if not os.path.isfile(filepath):
            self.logMessage(self.tr('Error! Dysmal param file does not exist: ')+filepath)
            return
        self.deselectParamFile()
        self.LineEditParamFile.setText(filepath)
        self.LineEditParamFile.setFocus()
        self.readParamFile(filepath)

    def deselectParamFile(self):
        if self.DysmalPyParams is not None:
            # deselect the current param file, pop up a window to ask if the user wants to save it
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Question)
            msgBox.setText("Parameters changed! Do you want to save current parameters?")
            msgBox.setWindowTitle("Save param file?")
            clickedButton = msgBox.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
            if clickedButton == QMessageBox.Save:
                self.saveParamFile()
        self.LineEditParamFile.setText('')
    
    def readParamFile(self, filepath):
        #
        #if self.DysmalPyParamFile is not None:
        #    self.logMessage(self.tr('Selecting Dysmal param file: ')+str(filepath))
        #    if filepath == self.DysmalPyParamFile:
        #        self.logger.debug('The selected file is the same as current file. Do nothing.')
        #        return
        #
        self.logMessage(self.tr('Reading Dysmal param file: ')+str(filepath))
        self.setButtonsEnabledDisabled()
        errormessages = []
        params = utils_io.read_fitting_params(fname=filepath) #<TODO><DEBUG>#
        try:
            params = utils_io.read_fitting_params(fname=filepath)
        except Exception as err:
            params = None
            errormessages.append(str(err))
        if params is None:
            self.logMessage(self.tr('Error! Failed to load the Dysmal param file: ')+str(filepath))
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText(self.tr('Error! Failed to load the Dysmal param file: ')+'\n'+str(filepath)+'\n'+self.tr('Please check your Dysmal params file and open again.'))
            msgBox.setInformativeText(self.tr('Error messages:')+'\n'+'\n'.join(errormessages))
            msgBox.setWindowTitle("Failed to load the Dysmal param file")
            msgBox.setStandardButtons(QMessageBox.Ok)
            #msgBox.buttonClicked.connect(msgButtonClick)
            msgBox.exec()
            return
        # 
        if 'oversample' in params:
            if params['oversample'] is None or str(params['oversample']) == 'None':
                params['oversample'] = 1 # a simple fix, 20230804
        #
        self.DysmalPyParamFile = filepath
        self.DysmalPyParams = params
        for keyname in ['fdata', 'fdata_flux', 'fdata_ferr', 'fdata_vel', 'fdata_verr', 'fdata_disp', 'fdata_derr', 'fdata_mask', 'fdata_cube', 'fdata_err']:
            if keyname not in params:
                self.LineEditDataParamsDict[keyname].setText('', blocksignal=True)
                self.LineEditDataParamsDict[keyname].setEnabled(False)
        #
        if 'datadir' not in self.DysmalPyParams:
            self.DysmalPyParams['datadir'] = None
        if self.DysmalPyParams['datadir'] is not None and self.DysmalPyParams['datadir'] != '':
            self.DysmalPyParams['datadir'] = ensure_path_trailing_slash(self.DysmalPyParams['datadir'])
        if 'outdir' not in self.DysmalPyParams:
            self.DysmalPyParams['outdir'] = None
        if self.DysmalPyParams['outdir'] is not None and self.DysmalPyParams['outdir'] != '':
            self.DysmalPyParams['outdir'] = ensure_path_trailing_slash(self.DysmalPyParams['outdir'])
        #
        #self.logger.debug("self.DysmalPyParams['datadir']: "+str(self.DysmalPyParams['datadir']))
        #self.logger.debug("self.DysmalPyParams['outdir']: "+str(self.DysmalPyParams['outdir']))
        #
        # fill in GUI LineEdit
        for key in self.LineEditDataParamsDict:
            if key in self.DysmalPyParams:
                #self.logMessage('Updating LineEditDataParamsDict[%r] = %s'%(key, self.DysmalPyParams[key]))
                self.LineEditDataParamsDict[key].setText(self.DysmalPyParams[key], blocksignal=True)
        for key in self.LineEditLensingParamsDict:
            if key in self.DysmalPyParams:
                #self.logMessage('Updating LineEditLensingParamsDict[%r] = %s'%(key, self.DysmalPyParams[key]))
                self.LineEditLensingParamsDict[key].setText(self.DysmalPyParams[key], blocksignal=True)
        for i in range(len(self.LineEditModelParamsDicts)):
            for key in self.LineEditModelParamsDicts[i]:
                if key in self.DysmalPyParams:
                    #self.logMessage('Updating LineEditModelParamsDicts[%r] = %s'%(key, self.DysmalPyParams[key]))
                    self.LineEditModelParamsDicts[i][key].setText(self.DysmalPyParams[key], blocksignal=True)
        for key in self.CheckBoxModelParamsDict:
            if key in self.DysmalPyParams:
                self.logMessage('Updating CheckBoxModelParamsDict[%r] = %s'%(key, self.DysmalPyParams[key]))
                self.CheckBoxModelParamsDict[key].setChecked(self.DysmalPyParams[key]) # , blocksignal=True
                # if checked state not changed, manually run a changed call
                if self.CheckBoxModelParamsDict[key].text() == self.CheckBoxModelParamsDict[key].default():
                    if key == 'zheight_tied':
                        self.onZheightTiedCheckStateChangedCall(0)
                    elif key == 'mvirial_tied':
                        self.onMvirialTiedCheckStateChangedCall(0)
                    elif key == 'fdm_tied':
                        self.onFdmTiedCheckStateChangedCall(0)
            # overwrite checkbox depends on the 'overwrite' key
            elif key == '__overwriting__' and 'overwrite' in self.DysmalPyParams:
                self.logMessage('Updating CheckBoxModelParamsDict[%r] = %s'%(key, self.DysmalPyParams['overwrite']))
                self.CheckBoxModelParamsDict[key].setChecked(self.DysmalPyParams['overwrite']) # , blocksignal=True
            # gauss_extract checkbox depends on the 'moment_calc' key
            elif key == '__gauss_extract__' and 'moment_calc' in self.DysmalPyParams:
                self.logMessage('Updating CheckBoxModelParamsDict[%r] = %s'%(key, self.DysmalPyParams['moment_calc']==False))
                self.CheckBoxModelParamsDict[key].setChecked(self.DysmalPyParams['moment_calc']==False) # , blocksignal=True
            # gauss_extract_with_c checkbox depends on the 'moment_calc' key
            elif key == 'gauss_extract_with_c' and 'moment_calc' in self.DysmalPyParams:
                self.logMessage('Updating CheckBoxModelParamsDict[%r] = %s'%(key, self.DysmalPyParams['moment_calc']==False))
                self.CheckBoxModelParamsDict[key].setChecked(self.DysmalPyParams['moment_calc']==False) # , blocksignal=True
                self.DysmalPyParams['gauss_extract_with_c'] = self.CheckBoxModelParamsDict[key].keyvalue()
                # 'gauss_extract_with_c' is enabled in default in our GUI, 
                # if it is not specified in the params file, 
                # but 'moment_calc' is specified in params and is set to False. 
        #
        # check data file data directory, see if we can proceed to load existing best fit result
        errormessages = []
        if self.DysmalPyParams['datadir'] is None or self.DysmalPyParams['datadir'] == '':
            pass # errormessages.append('datadir is None')
        elif not os.path.isdir(self.DysmalPyParams['datadir']):
            errormessages.append('datadir is not found on disk: '+self.DysmalPyParams['datadir'])
        if self.DysmalPyParams['outdir'] is None or self.DysmalPyParams['outdir'] == '':
            errormessages.append('outdir is None')
        hasdata = False
        for keyname in ['fdata', 'fdata_flux', 'fdata_ferr', 'fdata_vel', 'fdata_verr', 'fdata_disp', 'fdata_derr', 'fdata_mask', 'fdata_cube', 'fdata_err']:
            if keyname in self.DysmalPyParams:
                hasdata = True
                if self.DysmalPyParams['datadir'] is None or self.DysmalPyParams['datadir'] == '':
                    filepath_check = self.DysmalPyParams[keyname]
                else:
                    filepath_check = os.path.join(self.DysmalPyParams['datadir'], self.DysmalPyParams[keyname])
                if not os.path.isfile(filepath_check):
                    errormessages.append(keyname+' is not found on disk: '+self.DysmalPyParams[keyname])
        if not hasdata:
            errormessages.append('No fdata* key set!')
        #
        if not ('moment_calc' in self.DysmalPyParams):
            errormessages.append('No moment_calc key set!')
        #
        if ('fitflux' in self.DysmalPyParams):
            if (self.DysmalPyParams['fitflux']):
                if not ('fdata_flux' in self.DysmalPyParams and 'fdata_ferr' in self.DysmalPyParams):
                    errormessages.append('No fdata_flux and fdata_ferr keys when fitflux is set!')
        # 
        if ('galID' in self.DysmalPyParams):
            self.DysmalPyParams['galID'] = str(self.DysmalPyParams['galID'])
        #
        if len(errormessages) > 0:
            self.logMessage(self.tr('Found errors in the Dysmal param file: ')+str(filepath))
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText(self.tr('Found errors in the Dysmal param file: ')+'\n'+str(filepath)+'\n'+self.tr('Please check your Dysmal params file and correct the errors.'))
            msgBox.setInformativeText(self.tr('Error messages:')+'\n'+'\n'.join(errormessages))
            msgBox.setWindowTitle("Found errors in the Dysmal param file")
            msgBox.setStandardButtons(QMessageBox.Ok)
            #msgBox.buttonClicked.connect(msgButtonClick)
            msgBox.exec()
            self.setButtonsEnabledDisabled()
            return
        #
        self.logMessage(self.tr('Successfully loaded Dysmal param file: ')+str(self.DysmalPyParamFile))
        #
        # check if there are already best fit file
        output_pickle_file = os.path.join(self.DysmalPyParams['outdir'], self.DysmalPyParams['galID']+'_'+self.DysmalPyParams['fit_method']+'_results.pickle')
        if os.path.exists(output_pickle_file):
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Question)
            msgBox.setText("Would you like to load previous bestfits?")
            msgBox.setWindowTitle("Load previous bestfits?")
            msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            clickedButton = msgBox.exec()
            if clickedButton == QMessageBox.Yes:
                self.fitDataAsync(doFit=False, clearFitResults=False)
        self.setButtonsEnabledDisabled()
    
    def writeOneLineToParamFile(self, fp, keyname, keyvalue, keycomment, datatype, listtype):
        if keyname.find('NULL')>=0 or keyname.startswith('__'):
            return
        if keycomment is None:
            keycomment = ''
        #ensure_path_trailing_slash:
        #if not keyvalue.endswith(os.sep):
        #    keyvalue += os.sep
        if keyvalue == '' and datatype in [int, float] and listtype is not list:
            keyvalue = 'None'
        if listtype is list:
            #keyvalue = re.sub(r'[, ]+', r' ', re.sub(r'^\[(.*)\]$', r'\1', str(keyvalue).strip()))
            #keyvalue = ' '.join([str(t) for t in eval(str(keyvalue))])
            #keyvalue = ' '.join([str(t) for t in (re.sub(r'[\[\],]', r' ', str(keyvalue))).strip().split()])
            keyvalue = ' '.join([str(t) for t in (re.sub(r'[\[\]\'\",]', r' ', str(keyvalue))).strip().split()])
        if keycomment.strip() != '':
            fp.write('{:<26} {:<62} {} {} \n'.format(keyname+',', keyvalue, '#', keycomment))
        else:
            fp.write('{:<26} {:<62} \n'.format(keyname+',', keyvalue))
    
    def saveParamFile(self, filepath=None):
        if self.DysmalPyParams is None:
            self.logMessage(self.tr('Error! DysmalPyParams not loaded!'))
            return
        if filepath is None or filepath == '':
            filepath, _ = QFileDialog.getSaveFileName(self, self.tr('Saving Dysmal params file'), self.DefaultDirectory, self.tr('Dysmal Params File (*.params *.*)')) # QFileDialog::DontConfirmOverwrite
            if filepath is None or filepath == '':
                self.logMessage(self.tr('No file selected.'))
                return
        if not filepath.endswith('.params'):
            filepath += '.params'
        if os.path.isfile(filepath):
            self.logger.debug('Found existing param file %r. Backing up as %r.'%(filepath, filepath+'.backup'))
            shutil.move(filepath, filepath+'.backup')
        with open(filepath, 'w') as fp:
            #self.writeOneLineToParamFile(fp, 'datadir', self.LineEditDataDirDisplayed.text(), self.LineEditDataDirDisplayed.toolTip())
            #keyname = 'fdata'
            #if 'fdata' in self.DysmalPyParams:
            #    keyname = 'fdata'
            #elif 'fdata_vel' in self.DysmalPyParams:
            #    keyname = 'fdata_vel'
            #elif 'fdata_cube' in self.DysmalPyParams:
            #    keyname = 'fdata_cube'
            #self.writeOneLineToParamFile(fp, keyname, self.LineEditDataFileDisplayed.text(), self.LineEditDataFileDisplayed.toolTip())
            #self.writeOneLineToParamFile(fp, 'outdir', self.LineEditOutDirDisplayed.text(), self.LineEditOutDirDisplayed.toolTip())
            for key in self.LineEditDataParamsDict:
                if self.LineEditDataParamsDict[key].isEnabled():
                    self.writeOneLineToParamFile(fp, key, \
                                                 self.LineEditDataParamsDict[key].text(),\
                                                 self.LineEditDataParamsDict[key].toolTip(),\
                                                 self.LineEditDataParamsDict[key].dataType(),\
                                                 self.LineEditDataParamsDict[key].listType()\
                                                 )
            for key in self.LineEditLensingParamsDict:
                if self.LineEditLensingParamsDict[key].isEnabled():
                    self.writeOneLineToParamFile(fp, key, \
                                                 self.LineEditLensingParamsDict[key].text(),\
                                                 self.LineEditLensingParamsDict[key].toolTip(),\
                                                 self.LineEditLensingParamsDict[key].dataType(),\
                                                 self.LineEditLensingParamsDict[key].listType()\
                                                 )
            for i in range(len(self.LineEditModelParamsDicts)):
                for key in self.LineEditModelParamsDicts[i]:
                    if self.LineEditModelParamsDicts[i][key].isEnabled():
                        self.writeOneLineToParamFile(fp, key, \
                                                     self.LineEditModelParamsDicts[i][key].text(),\
                                                     self.LineEditModelParamsDicts[i][key].toolTip(),\
                                                     self.LineEditModelParamsDicts[i][key].dataType(),\
                                                     self.LineEditModelParamsDicts[i][key].listType()\
                                                     )
            for key in self.CheckBoxModelParamsDict:
                if self.CheckBoxModelParamsDict[key].isEnabled():
                    self.writeOneLineToParamFile(fp, key, \
                                                 self.CheckBoxModelParamsDict[key].text(),\
                                                 self.CheckBoxModelParamsDict[key].toolTip(),\
                                                 self.CheckBoxModelParamsDict[key].dataType(),\
                                                 self.CheckBoxModelParamsDict[key].listType()\
                                                 )
            if 'components_list' in self.DysmalPyParams:
                key = 'components_list'
                self.writeOneLineToParamFile(fp, key, \
                                             self.DysmalPyParams[key],\
                                             '',\
                                             str,\
                                             list\
                                             )
        self.logMessage(self.tr('Saved parameters to param file: ')+str(filepath))
    
    @pyqtSlot()
    def onSaveParamFileCall(self, filepath=None):
        self.saveParamFile(filepath=None)
    
    @pyqtSlot()
    def onSaveModelCubeCall(self, filepath=None):
        if self.DysmalPyFittingTower.model_cube is None:
            self.logMessage(self.tr('Error! No valid model cube! Could not save to FITS file.'))
            return
        self.saveFitsFile(data=self.DysmalPyFittingTower.model_cube_data_array)
    
    @pyqtSlot()
    def onSaveModelFluxCall(self):
        if self.DysmalPyFittingTower.model_flux_map is None:
            self.logMessage(self.tr('Error! No valid model flux map! Could not save to FITS file.'))
            return
        self.saveFitsFile(data=self.DysmalPyFittingTower.model_flux_map)
    
    @pyqtSlot()
    def onSaveModelVelCall(self):
        if self.DysmalPyFittingTower.model_vel_map is None:
            self.logMessage(self.tr('Error! No valid model vel map! Could not save to FITS file.'))
            return
        self.saveFitsFile(data=self.DysmalPyFittingTower.model_vel_map)
    
    @pyqtSlot()
    def onSaveModelVdispCall(self):
        if self.DysmalPyFittingTower.model_disp_map is None:
            self.logMessage(self.tr('Error! No valid model disp map! Could not save to FITS file.'))
            return
        self.saveFitsFile(data=self.DysmalPyFittingTower.model_disp_map)
    
    def hasLensingTransformer(self):
        """Check if a lensing transformer exists in the DysmalPyFittingTower class.
        """
        return self.DysmalPyFittingTower.lensing_transformer is not None
    
    def clearLensingTransformer(self):
        """Clear a lensing transformer in the DysmalPyFittingTower class.
        """
        self.DysmalPyFittingTower.BaseQueue.put(
            ( 'command', 'clear_lensing_transformer' )
        )
    
    @pyqtSlot()
    def onSaveModelFilesCall(self, filepath=None):
        """Save all model cube, moment maps and figures.
        """
        errormessages = []
        if self.DysmalPyFittingTower.model_cube_data_array is None:
            errormessages.append(self.tr('Error! No valid model cube! Could not save the FITS file.'))
        if self.DysmalPyFittingTower.model_flux_map is None:
            errormessages.append(self.tr('Error! No valid model flux map! Could not save the FITS file.'))
        if self.DysmalPyFittingTower.model_vel_map is None:
            errormessages.append(self.tr('Error! No valid model vel map! Could not save the FITS file.'))
        if self.DysmalPyFittingTower.model_disp_map is None:
            errormessages.append(self.tr('Error! No valid model disp map! Could not save the FITS file.'))
        if len(errormessages) != 0:
            return
        #
        defaultpath = self.DefaultDirectory
        if 'outdir' in self.DysmalPyParams:
            if self.DysmalPyParams['outdir'] is not None:
                defaultpath = self.DysmalPyParams['outdir']
        filepath, _ = QFileDialog.getSaveFileName(self, self.tr('Saving model files'), defaultpath, self.tr('Base name for output (*.*)'))
        if filepath is None or filepath == '':
            self.logMessage(self.tr('No file selected. No file saved.'))
            return
        filepath = re.sub(r'^(.*)\.fits$', r'\1', filepath)
        self.saveFitsFile(data=self.DysmalPyFittingTower.model_cube_data_array, 
                          header=self.DysmalPyFittingTower.model_cube_header_info, 
                          filepath=filepath+'_model_cube.fits')
        self.saveFitsFile(data=self.DysmalPyFittingTower.model_flux_map, filepath=filepath+'_model_flux_map.fits')
        self.saveFitsFile(data=self.DysmalPyFittingTower.model_vel_map, filepath=filepath+'_model_vel_map.fits')
        self.saveFitsFile(data=self.DysmalPyFittingTower.model_disp_map, filepath=filepath+'_model_disp_map.fits')
        self.saveFitsFile(data=self.DysmalPyFittingTower.residual_flux_map, filepath=filepath+'_residual_flux_map.fits')
        self.saveFitsFile(data=self.DysmalPyFittingTower.residual_vel_map, filepath=filepath+'_residual_vel_map.fits')
        self.saveFitsFile(data=self.DysmalPyFittingTower.residual_disp_map, filepath=filepath+'_residual_disp_map.fits')
        self.saveFitsFile(data=self.DysmalPyFittingTower.data_flux_map, filepath=filepath+'_data_flux_map.fits')
        self.saveFitsFile(data=self.DysmalPyFittingTower.data_vel_map, filepath=filepath+'_data_vel_map.fits')
        self.saveFitsFile(data=self.DysmalPyFittingTower.data_disp_map, filepath=filepath+'_data_disp_map.fits')
        self.saveFitsFile(data=self.DysmalPyFittingTower.data_mask_map, filepath=filepath+'_data_mask_map.fits')
        self.saveJsonFile(data=self.DysmalPyFittingTower.model_flux_curve, filepath=filepath+'_model_flux_curve.json')
        self.saveJsonFile(data=self.DysmalPyFittingTower.model_vel_curve, filepath=filepath+'_model_vel_curve.json')
        self.saveJsonFile(data=self.DysmalPyFittingTower.model_disp_curve, filepath=filepath+'_model_disp_curve.json')
        self.SpecViewerA1.fig.savefig(filepath+'_model_flux_curve.pdf', dpi=300, transparent=True)
        self.SpecViewerA2.fig.savefig(filepath+'_model_vel_curve.pdf', dpi=300, transparent=True)
        self.SpecViewerA3.fig.savefig(filepath+'_model_disp_curve.pdf', dpi=300, transparent=True)
        self.ImageViewerA1.fig.savefig(filepath+'_data_flux_map.pdf', dpi=300, transparent=True)
        self.ImageViewerA2.fig.savefig(filepath+'_data_vel_map.pdf', dpi=300, transparent=True)
        self.ImageViewerA3.fig.savefig(filepath+'_data_disp_map.pdf', dpi=300, transparent=True)
        self.ImageViewerB1.fig.savefig(filepath+'_model_flux_map.pdf', dpi=300, transparent=True)
        self.ImageViewerB2.fig.savefig(filepath+'_model_vel_map.pdf', dpi=300, transparent=True)
        self.ImageViewerB3.fig.savefig(filepath+'_model_disp_map.pdf', dpi=300, transparent=True)
        self.ImageViewerC1.fig.savefig(filepath+'_residual_flux_map.pdf', dpi=300, transparent=True)
        self.ImageViewerC2.fig.savefig(filepath+'_residual_vel_map.pdf', dpi=300, transparent=True)
        self.ImageViewerC3.fig.savefig(filepath+'_residual_disp_map.pdf', dpi=300, transparent=True)
        if self.DysmalPyFittingTower.lensing_transformer_image_plane_data_cube is not None:
            self.saveFitsFile(data=self.DysmalPyFittingTower.lensing_transformer_image_plane_data_cube,
                              header=self.DysmalPyFittingTower.lensing_transformer_image_plane_data_info,
                              filepath=filepath+'_lensing_image_plane_data_cube.fits')
        if self.DysmalPyFittingTower.lensing_transformer_source_plane_data_cube is not None:
            self.saveFitsFile(data=self.DysmalPyFittingTower.lensing_transformer_source_plane_data_cube,
                              header=self.DysmalPyFittingTower.lensing_transformer_source_plane_data_info,
                              filepath=filepath+'_lensing_source_plane_data_cube.fits')
        self.saveParamFile(filepath=filepath+'.params')
    
    def saveFitsFile(self, data, header=None, filepath=None):
        if data is None:
            return
        if hasattr(data, 'dtype'):
            if data.dtype == bool:
                data = data.astype(int) # if data dtype is boolean, convert to int so as to save as a FITS file.
        if filepath is None or filepath == '':
            filepath, _ = QFileDialog.getSaveFileName(self, self.tr('Saving as FITS file'), self.DefaultDirectory, self.tr('FITS file (*.fits)'))
            if filepath is None or filepath == '':
                self.logMessage(self.tr('No file selected. No file saved.'))
                return
        if os.path.isfile(filepath):
            self.logger.debug('Found existing FITS file %r. Backing up as %r.'%(filepath, filepath+'.backup'))
            shutil.move(filepath, filepath+'.backup')
        if header is not None:
            if isinstance(header, (dict, OrderedDict)):
                header2 = copy.copy(header)
                header = fits.Header()
                for key in header2:
                    header[key] = header2[key]
            hdu = fits.PrimaryHDU(data=data, header=header)
        else:
            hdu = fits.PrimaryHDU(data=data)
        hdu.writeto(filepath)
        self.logMessage(self.tr('Saved to FITS file: ')+str(filepath))
    
    def saveJsonFile(self, data, filepath=None, sort_keys=True, indent=4):
        if data is None:
            return
        if filepath is None or filepath == '':
            filepath, _ = QFileDialog.getSaveFileName(self, self.tr('Saving as JSON file'), self.DefaultDirectory, self.tr('JSON file (*.json)'))
            if filepath is None or filepath == '':
                self.logMessage(self.tr('No file selected. No file saved.'))
                return
        if os.path.isfile(filepath):
            self.logger.debug('Found existing JSON file %r. Backing up as %r.'%(filepath, filepath+'.backup'))
            shutil.move(filepath, filepath+'.backup')
        with open(filepath, 'w') as fp:
            json.dump(data, fp, sort_keys=sort_keys, indent=indent, cls=NumpyEncoder) # write spwmap to json file
        self.logMessage(self.tr('Saved to JSON file: ')+str(filepath))
    
    def readJsonFile(self, filepath=None):
        if filepath is None or filepath == '':
            filepath, _ = QFileDialog.getOpenFileName(self, self.tr('Opening a JSON file'), self.DefaultDirectory, self.tr('JSON file (*.json)'))
            if filepath is None or filepath == '':
                self.logMessage(self.tr('No file selected.'))
                return None
        self.logMessage(self.tr('Reading JSON file: ')+str(filepath))
        with open(filepath, 'r') as fp:
            data = json.load(fp)
            data = ast.literal_eval(json.dumps(data)) # Removing uni-code chars
        return data
    
    @pyqtSlot()
    def onGenerateModelCubeCall(self):
        self.logger.debug('onGenerateModelCubeCall()')
        if self.checkDysmalPyParams():
            self.generateModelCubeAsync(clearFitResults=True)
    
    @pyqtSlot()
    def onGenerateMomentMapsCall(self):
        self.logger.debug('onGenerateMomentMapsCall()')
        if self.checkDysmalPyParams():
            self.generateMomentMapsAsync()
    
    @pyqtSlot()
    def onGenerateRotationCurvesCall(self):
        self.logger.debug('onGenerateRotationCurvesCall()')
        if self.checkDysmalPyParams():
            self.generateRotationCurvesAsync()
    
    @pyqtSlot()
    def onHideSlitCall(self):
        self.logger.debug('onHideSlitCall()')
        if self.checkDysmalPyParams():
            # hide/show the slit in the image viewer
            if self.DysmalPyFittingTower.data_flux_map is not None:
                if self.ImageViewerA1.slit is not None:
                    if self.ImageViewerA1.slit.get_visible() == True:
                        self.ImageViewerA1.hideSlit()
                        self.ButtonHideSlit.setText(self.tr('Show Slit'))
                    else:
                        self.ImageViewerA1.showSlit()
                        self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            if self.DysmalPyFittingTower.data_vel_map is not None:
                if self.ImageViewerA2.slit is not None:
                    if self.ImageViewerA2.slit.get_visible() == True:
                        self.ImageViewerA2.hideSlit()
                        self.ButtonHideSlit.setText(self.tr('Show Slit'))
                    else:
                        self.ImageViewerA2.showSlit()
                        self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            if self.DysmalPyFittingTower.data_disp_map is not None:
                if self.ImageViewerA3.slit is not None:
                    if self.ImageViewerA3.slit.get_visible() == True:
                        self.ImageViewerA3.hideSlit()
                        self.ButtonHideSlit.setText(self.tr('Show Slit'))
                    else:
                        self.ImageViewerA3.showSlit()
                        self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            #
            if self.DysmalPyFittingTower.model_flux_map is not None:
                if self.ImageViewerB1.slit is not None:
                    if self.ImageViewerB1.slit.get_visible() == True:
                        self.ImageViewerB1.hideSlit()
                        self.ButtonHideSlit.setText(self.tr('Show Slit'))
                    else:
                        self.ImageViewerB1.showSlit()
                        self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            if self.DysmalPyFittingTower.model_vel_map is not None:
                if self.ImageViewerB2.slit is not None:
                    if self.ImageViewerB2.slit.get_visible() == True:
                        self.ImageViewerB2.hideSlit()
                        self.ButtonHideSlit.setText(self.tr('Show Slit'))
                    else:
                        self.ImageViewerB2.showSlit()
                        self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            if self.DysmalPyFittingTower.model_disp_map is not None:
                if self.ImageViewerB3.slit is not None:
                    if self.ImageViewerB3.slit.get_visible() == True:
                        self.ImageViewerB3.hideSlit()
                        self.ButtonHideSlit.setText(self.tr('Show Slit'))
                    else:
                        self.ImageViewerB3.showSlit()
                        self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            #
            #if self.ButtonHideSlit.text() == self.tr('Show Slit'):
            #    self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            #else:
            #    self.ButtonHideSlit.setText(self.tr('Show Slit'))
    
    def plotDataMomentMaps(self):
        self.logMessage(self.tr('Computing and displaying data cube moment maps'))
        if self.DysmalPyFittingTower.data_flux_map is not None:
            self.ImageViewerA1.showImage(self.DysmalPyFittingTower.data_flux_map, with_colorbar=True, cmap='viridis')
        if self.DysmalPyFittingTower.data_vel_map is not None:
            self.ImageViewerA2.showImage(self.DysmalPyFittingTower.data_vel_map, with_colorbar=True) # , cmap='RdYlBu_r'
        if self.DysmalPyFittingTower.data_disp_map is not None:
            self.ImageViewerA3.showImage(self.DysmalPyFittingTower.data_disp_map, with_colorbar=True, cmap='plasma')
        self.logMessage(self.tr('Successfully computed and displayed data cube moment maps'))

    def plotModelMomentMaps(self):
        self.logMessage(self.tr('Computing and displaying model cube moment maps'))
        if self.DysmalPyFittingTower.model_flux_map is not None:
            self.ImageViewerB1.showImage(self.DysmalPyFittingTower.model_flux_map, with_colorbar=True, cmap='viridis')
        if self.DysmalPyFittingTower.model_vel_map is not None:
            self.ImageViewerB2.showImage(self.DysmalPyFittingTower.model_vel_map, with_colorbar=True) # , cmap='RdYlBu_r'
        if self.DysmalPyFittingTower.model_disp_map is not None:
            self.ImageViewerB3.showImage(self.DysmalPyFittingTower.model_disp_map, with_colorbar=True, cmap='plasma')
        if self.DysmalPyFittingTower.residual_flux_map is not None:
            self.ImageViewerC1.showImage(self.DysmalPyFittingTower.residual_flux_map, with_colorbar=True) # , cmap='RdYlBu_r'
        if self.DysmalPyFittingTower.residual_vel_map is not None:
            self.ImageViewerC2.showImage(self.DysmalPyFittingTower.residual_vel_map, with_colorbar=True) # , cmap='RdYlBu_r'
        if self.DysmalPyFittingTower.residual_disp_map is not None:
            self.ImageViewerC3.showImage(self.DysmalPyFittingTower.residual_disp_map, with_colorbar=True) # , cmap='RdYlBu_r'
        self.logMessage(self.tr('Successfully computed and displayed model cube moment maps'))
    
    def plotModelRotationCurves(self):
        self.logMessage(self.tr('Computing and displaying model rotation curve'))
        #
        clear_plot = True
        has_plot = False
        if self.DysmalPyFittingTower.data_flux_curve is not None:
            self.logger.debug('plotModelRotationCurves() self.SpecViewerA1.plotSpectrum self.DysmalPyFittingTower.data_flux_curve')
            self.SpecViewerA1.plotSpectrum(**(self.DysmalPyFittingTower.data_flux_curve), clear_plot=clear_plot, label='data', color='k', zorder=90)
            clear_plot = False
            has_plot = True
        if self.DysmalPyFittingTower.model_flux_curve is not None:
            self.logger.debug('plotModelRotationCurves() self.SpecViewerA1.plotSpectrum self.DysmalPyFittingTower.model_flux_curve')
            self.SpecViewerA1.plotSpectrum(**(self.DysmalPyFittingTower.model_flux_curve), clear_plot=clear_plot, label='model', color='red', zorder=99)
            clear_plot = False
            has_plot = True
        if has_plot:
            #self.SpecViewerA1.axes.set_title('Flux')
            self.SpecViewerA1.plotLegend(loc='upper left')
            # also show a slit as a line
            if self.DysmalPyFittingTower.data_flux_map is not None:
                self.ImageViewerA1.showSlit(self.getSlitShapeInPixel())
                self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            if self.DysmalPyFittingTower.model_flux_map is not None:
                self.ImageViewerB1.showSlit(self.getSlitShapeInPixel())
                self.ButtonHideSlit.setText(self.tr('Hide Slit'))
        #
        clear_plot = True
        has_plot = False
        if self.DysmalPyFittingTower.data_vel_curve is not None:
            self.logger.debug('plotModelRotationCurves() self.SpecViewerA2.plotSpectrum self.DysmalPyFittingTower.data_vel_curve')
            self.SpecViewerA2.plotSpectrum(**(self.DysmalPyFittingTower.data_vel_curve), clear_plot=clear_plot, label='data', color='k', zorder=90)
            clear_plot = False
            has_plot = True
        if self.DysmalPyFittingTower.model_vel_curve is not None:
            self.logger.debug('plotModelRotationCurves() self.SpecViewerA2.plotSpectrum self.DysmalPyFittingTower.model_vel_curve')
            self.SpecViewerA2.plotSpectrum(**(self.DysmalPyFittingTower.model_vel_curve), clear_plot=clear_plot, label='model', color='red', zorder=99)
            clear_plot = False
            has_plot = True
        if has_plot:
            #self.SpecViewerA2.axes.set_title('Vel')
            #self.SpecViewerA2.axes.legend(loc='upper left') # 20250827?
            #self.SpecViewerA2.draw() # 20250827?
            self.SpecViewerA2.plotLegend(loc='upper left')
            # also show a slit as a line
            if self.DysmalPyFittingTower.data_vel_map is not None:
                self.ImageViewerA2.showSlit(self.getSlitShapeInPixel())
                self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            if self.DysmalPyFittingTower.model_vel_map is not None:
                self.ImageViewerB2.showSlit(self.getSlitShapeInPixel())
                self.ButtonHideSlit.setText(self.tr('Hide Slit'))
        #
        clear_plot = True
        has_plot = False
        if self.DysmalPyFittingTower.data_disp_curve is not None:
            self.logger.debug('plotModelRotationCurves() self.SpecViewerA3.plotSpectrum self.DysmalPyFittingTower.data_disp_curve')
            self.SpecViewerA3.plotSpectrum(**(self.DysmalPyFittingTower.data_disp_curve), clear_plot=clear_plot, label='data', color='k', zorder=90)
            clear_plot = False
            has_plot = True
        if self.DysmalPyFittingTower.model_disp_curve is not None:
            self.logger.debug('plotModelRotationCurves() self.SpecViewerA3.plotSpectrum self.DysmalPyFittingTower.model_disp_curve')
            self.SpecViewerA3.plotSpectrum(**(self.DysmalPyFittingTower.model_disp_curve), clear_plot=clear_plot, label='model', color='red', zorder=99)
            clear_plot = False
            has_plot = True
        if has_plot:
            #self.SpecViewerA3.axes.set_title('Vdisp')
            #self.SpecViewerA3.axes.legend(loc='upper left') # 20250827?
            #self.SpecViewerA3.draw() # 20250827?
            self.SpecViewerA3.plotLegend(loc='upper left')
            # also show a slit as a line
            if self.DysmalPyFittingTower.data_disp_map is not None:
                self.ImageViewerA3.showSlit(self.getSlitShapeInPixel())
                self.ButtonHideSlit.setText(self.tr('Hide Slit'))
            if self.DysmalPyFittingTower.model_disp_map is not None:
                self.ImageViewerB3.showSlit(self.getSlitShapeInPixel())
                self.ButtonHideSlit.setText(self.tr('Hide Slit'))
        #
        self.logMessage(self.tr('Successfully computed and displayed model rotation curve'))
    
    @pyqtSlot(PixCoord)
    def selectPixelInImageViewers(self, pixel:PixCoord = None):
        if pixel is not None:
            #
            if self.DysmalPyFittingTower.data_flux_map is not None:
                self.ImageViewerA1.blockSignals(True)
            if self.DysmalPyFittingTower.data_vel_map is not None:
                self.ImageViewerA2.blockSignals(True)
            if self.DysmalPyFittingTower.data_disp_map is not None:
                self.ImageViewerA3.blockSignals(True)
            if self.DysmalPyFittingTower.model_flux_map is not None:
                self.ImageViewerB1.blockSignals(True)
            if self.DysmalPyFittingTower.model_vel_map is not None:
                self.ImageViewerB2.blockSignals(True)
            if self.DysmalPyFittingTower.model_disp_map is not None:
                self.ImageViewerB3.blockSignals(True)
            if self.DysmalPyFittingTower.residual_flux_map is not None:
                self.ImageViewerC1.blockSignals(True)
            if self.DysmalPyFittingTower.residual_vel_map is not None:
                self.ImageViewerC2.blockSignals(True)
            if self.DysmalPyFittingTower.residual_disp_map is not None:
                self.ImageViewerC3.blockSignals(True)
            #
            if self.DysmalPyFittingTower.data_flux_map is not None:
                self.ImageViewerA1.setSelectedPixel(pixel.x, pixel.y)
            if self.DysmalPyFittingTower.data_vel_map is not None:
                self.ImageViewerA2.setSelectedPixel(pixel.x, pixel.y)
            if self.DysmalPyFittingTower.data_disp_map is not None:
                self.ImageViewerA3.setSelectedPixel(pixel.x, pixel.y)
            if self.DysmalPyFittingTower.model_flux_map is not None:
                self.ImageViewerB1.setSelectedPixel(pixel.x, pixel.y)
            if self.DysmalPyFittingTower.model_vel_map is not None:
                self.ImageViewerB2.setSelectedPixel(pixel.x, pixel.y)
            if self.DysmalPyFittingTower.model_disp_map is not None:
                self.ImageViewerB3.setSelectedPixel(pixel.x, pixel.y)
            if self.DysmalPyFittingTower.residual_flux_map is not None:
                self.ImageViewerC1.setSelectedPixel(pixel.x, pixel.y)
            if self.DysmalPyFittingTower.residual_vel_map is not None:
                self.ImageViewerC2.setSelectedPixel(pixel.x, pixel.y)
            if self.DysmalPyFittingTower.residual_disp_map is not None:
                self.ImageViewerC3.setSelectedPixel(pixel.x, pixel.y)
            #
            if self.DysmalPyFittingTower.data_flux_map is not None:
                self.ImageViewerA1.blockSignals(False)
            if self.DysmalPyFittingTower.data_vel_map is not None:
                self.ImageViewerA2.blockSignals(False)
            if self.DysmalPyFittingTower.data_disp_map is not None:
                self.ImageViewerA3.blockSignals(False)
            if self.DysmalPyFittingTower.model_flux_map is not None:
                self.ImageViewerB1.blockSignals(False)
            if self.DysmalPyFittingTower.model_vel_map is not None:
                self.ImageViewerB2.blockSignals(False)
            if self.DysmalPyFittingTower.model_disp_map is not None:
                self.ImageViewerB3.blockSignals(False)
            if self.DysmalPyFittingTower.residual_flux_map is not None:
                self.ImageViewerC1.blockSignals(False)
            if self.DysmalPyFittingTower.residual_vel_map is not None:
                self.ImageViewerC2.blockSignals(False)
            if self.DysmalPyFittingTower.residual_disp_map is not None:
                self.ImageViewerC3.blockSignals(False)
    
    @pyqtSlot(int)
    def selectChannelInSpecViewers(self, ichan:int = None):
        if ichan is not None:
            #
            if (self.DysmalPyFittingTower.data_flux_curve is not None or self.DysmalPyFittingTower.model_flux_curve is not None):
                self.SpecViewerA1.blockSignals(True)
            if (self.DysmalPyFittingTower.data_vel_curve is not None or self.DysmalPyFittingTower.model_vel_curve is not None):
                self.SpecViewerA2.blockSignals(True)
            if (self.DysmalPyFittingTower.data_disp_curve is not None or self.DysmalPyFittingTower.model_disp_curve is not None):
                self.SpecViewerA3.blockSignals(True)
            #
            if (self.DysmalPyFittingTower.data_flux_curve is not None or self.DysmalPyFittingTower.model_flux_curve is not None):
                self.SpecViewerA1.setSelectedChannel(ichan)
            if (self.DysmalPyFittingTower.data_vel_curve is not None or self.DysmalPyFittingTower.model_vel_curve is not None):
                self.SpecViewerA2.setSelectedChannel(ichan)
            if (self.DysmalPyFittingTower.data_disp_curve is not None or self.DysmalPyFittingTower.model_disp_curve is not None):
                self.SpecViewerA3.setSelectedChannel(ichan)
            #
            if (self.DysmalPyFittingTower.data_flux_curve is not None or self.DysmalPyFittingTower.model_flux_curve is not None):
                self.SpecViewerA1.blockSignals(False)
            if (self.DysmalPyFittingTower.data_vel_curve is not None or self.DysmalPyFittingTower.model_vel_curve is not None):
                self.SpecViewerA2.blockSignals(False)
            if (self.DysmalPyFittingTower.data_disp_curve is not None or self.DysmalPyFittingTower.model_disp_curve is not None):
                self.SpecViewerA3.blockSignals(False)
    
    @pyqtSlot()
    def onInitRandomParamsCall(self):
        if self.DysmalPyParamFile is not None and self.DysmalPyParamFile != '':
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Question)
            msgBox.setText("Parameters exist! Do you want to re-initialize and override current parameters?")
            msgBox.setWindowTitle("Re-initializing and overriding parameters?")
            msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
            clickedButton = msgBox.exec()
            if clickedButton != QMessageBox.Yes:
                return
        self.initRandomParams()
    
    def initRandomParams(self):
        self.logMessage(self.tr('Initializing DysmalPyParams with random values...'))
        # calling the setText() function of these QWidgetForParamInput
        # will invoke onLineEditTextChangedCall/onCheckBoxStateChangedCall/onComboBoxIndexChangedCall
        # because in class QWidgetForParamInput the signal is connected as
        # `self.LineEditWidget.textChanged.connect(self.onLineEditTextChangedCall)`
        # `self.CheckBoxWidget.stateChanged.connect(self.onCheckBoxStateChangedCall)`
        # `self.ComboBoxWidget.currentIndexChanged.connect(self.onComboBoxIndexChangedCall)`
        if self.LineEditDataParamsDict['galID'].text() == '':
            self.LineEditDataParamsDict['galID'].setText('galID')
        if self.LineEditDataParamsDict['z'].text() == '':
            self.LineEditDataParamsDict['z'].setText(str(np.round(np.random.uniform(low=0.5, high=3.0, size=(1))[0], 3)))
        if self.LineEditDataParamsDict['aperture_radius'].text() == '':
            self.LineEditDataParamsDict['aperture_radius'].setText('0.2')
        if self.LineEditDataParamsDict['linked_posteriors'].text() == '':
            self.LineEditDataParamsDict['linked_posteriors'].setText("['total_mass', 'r_eff_disk', 'bt', 'fdm', 'sigma0']")
        if self.LineEditDataParamsDict['pixscale'].text() == '':
            self.LineEditDataParamsDict['pixscale'].setText('0.05')
        if self.LineEditDataParamsDict['fov_npix'].text() == '':
            self.LineEditDataParamsDict['fov_npix'].setText('100')
        if self.LineEditDataParamsDict['spec_type'].text() == '':
            self.LineEditDataParamsDict['spec_type'].setText('velocity')
        if self.LineEditDataParamsDict['spec_start'].text() == '':
            self.LineEditDataParamsDict['spec_start'].setText('-1000.')
        if self.LineEditDataParamsDict['spec_step'].text() == '':
            self.LineEditDataParamsDict['spec_step'].setText('20.')
        if self.LineEditDataParamsDict['nspec'].text() == '':
            self.LineEditDataParamsDict['nspec'].setText('100')
        if self.LineEditDataParamsDict['sig_inst_res'].text() == '':
            self.LineEditDataParamsDict['sig_inst_res'].setText('5.')
        if self.LineEditDataParamsDict['psf_fwhm'].text() == '':
            self.LineEditDataParamsDict['psf_fwhm'].setText('0.3') # arcsec # superceded by psf_fwhm_major, psf_fwhm_minor
        if self.LineEditDataParamsDict['psf_fwhm_major'].text() == '':
            self.LineEditDataParamsDict['psf_fwhm_major'].setText('0.3') # arcsec
        if self.LineEditDataParamsDict['psf_fwhm_minor'].text() == '':
            self.LineEditDataParamsDict['psf_fwhm_minor'].setText('0.3') # arcsec
        if self.LineEditDataParamsDict['psf_PA'].text() == '':
            self.LineEditDataParamsDict['psf_PA'].setText('0.0') # degrees
        if self.LineEditDataParamsDict['slit_width'].text() == '':
            self.LineEditDataParamsDict['slit_width'].setText('0.5') # arcsec
        #if self.LineEditDataParamsDict['slit_pa'].text() == '':
        #    self.LineEditDataParamsDict['slit_pa'].setText(str(np.round(np.random.uniform(low=0.0, high=180.0, size=(1))[0], 3)))
        #
        self.LineEditModelParamsDictForBulgeDisk['total_mass'].setText(str(np.round(np.random.uniform(low=10.0, high=12.0, size=(1))[0], 3)))
        self.LineEditModelParamsDictForBulgeDisk['bt'].setText(str(np.round(np.random.uniform(low=0.0, high=1.0, size=(1))[0], 3)))
        self.LineEditModelParamsDictForBulgeDisk['r_eff_disk'].setText(str(np.round(np.random.uniform(low=5.0, high=27.0, size=(1))[0], 3)))
        self.LineEditModelParamsDictForBulgeDisk['n_disk'].setText(str(np.round(np.random.uniform(low=1.0, high=4.0, size=(1))[0], 3)))
        self.LineEditModelParamsDictForBulgeDisk['r_eff_bulge'].setText(str(np.round(np.random.uniform(low=1.0, high=4.5, size=(1))[0], 3))) # must be smaller than r_eff_disk and 5 kpc.
        self.LineEditModelParamsDictForBulgeDisk['n_bulge'].setText(str(np.round(np.random.uniform(low=2.0, high=4.0, size=(1))[0], 3)))
        self.LineEditModelParamsDictForDispersion['sigma0'].setText(str(np.round(np.random.uniform(low=10.0, high=100.0, size=(1))[0], 3)))
        self.LineEditModelParamsDictForZHeight['sigmaz'].setText(str(np.round(np.random.uniform(low=0.1, high=0.5, size=(1))[0], 3)))
        self.LineEditModelParamsDictForGeometry['inc'].setText(str(np.round(np.random.uniform(low=15.0, high=65.0, size=(1))[0], 3)))
        self.LineEditModelParamsDictForGeometry['pa'].setText(str(np.round(np.random.uniform(low=-180.0, high=180.0, size=(1))[0], 3)))
        self.LineEditDataParamsDict['slit_pa'].setText(self.LineEditModelParamsDictForGeometry['pa'].text())
        #
        self.LineEditModelParamsDictForDarkMatterHalo['mvirial'].setText(str(np.round(self.LineEditModelParamsDictForBulgeDisk['total_mass'].keyvalue()+np.random.uniform(low=0.5, high=1.0, size=(1))[0], 3)))
        self.LineEditModelParamsDictForDarkMatterHalo['halo_conc'].setText(str(np.round(np.random.uniform(low=3.0, high=6.0, size=(1))[0], 3)))
        self.LineEditModelParamsDictForDarkMatterHalo['fdm'].setText(str(np.round(np.random.uniform(low=0.1, high=0.6, size=(1))[0], 3)))
        # 
        for key in self.CheckBoxModelParamsDict:
            if self.CheckBoxModelParamsDict[key].isEnabled():
                #self.CheckBoxModelParamsDict[key].setChecked(self.CheckBoxModelParamsDict[key].keyvalue())
                #print(f'self.CheckBoxModelParamsDict[{key!r}].stateChanged.emit(1)')
                self.CheckBoxModelParamsDict[key].CheckBoxWidget.stateChanged.emit(1)
        #
        self.setButtonsEnabledDisabled()
        #
        self.logMessage(self.tr('Initialized DysmalPyParams with some values'))
    
    def setButtonsEnabledDisabled(self, allDisabled=None):
        has_gui = True
        has_par = (len(self.DysmalPyParams) != 0)
        has_validpar = False
        has_data = False
        has_bestfit = False
        has_model = False
        if allDisabled:
            has_gui = False
            has_par = False
        elif has_par:
            if np.any([t in self.DysmalPyParams for t in ['fdata', 'fdata_vel', 'fdata_cube']]):
                has_data = True
            if np.all([((t in self.DysmalPyParams) and (self.DysmalPyParams[t] is not None)) 
                       for t in [
                        'total_mass', 'bt', 'r_eff_disk', 
                        'n_disk', 'r_eff_bulge', 'n_bulge', 'sigma0', 'sigmaz', 'inc', 'pa',
                        'mvirial', 'halo_conc', 'fdm']
                      ]):
                has_validpar = True
                #print('has_validpar', has_validpar) # 20250728 DZLIU DEBUG
            if self.DysmalPyFittingTower is not None:
                if self.DysmalPyFittingTower.DysmalPyFitResultFile is not None:
                    if os.path.exists(self.DysmalPyFittingTower.DysmalPyFitResultFile):
                        has_bestfit = True
                # if self.DysmalPyFittingTower.model_cube is not None:
                if self.DysmalPyFittingTower.model_cube_data_array is not None:
                    has_model = True
        self.ButtonInitRandomParams.setEnabled(has_gui)
        self.ButtonOpenParamsFile.setEnabled(has_gui)
        self.ButtonFitData.setEnabled(has_data)
        self.ButtonLoadFittingResult.setEnabled(has_bestfit)
        self.ButtonGenerateModelCube.setEnabled(has_validpar)
        self.ButtonGenerateMomentMaps.setEnabled(has_model)
        self.ButtonGenerateRotationCurves.setEnabled(has_model)
        self.ButtonHideSlit.setEnabled(has_model)
        self.ButtonSaveParams.setEnabled(has_validpar)
        self.ButtonSaveModelFiles.setEnabled(has_model)
    
    @pyqtSlot()
    def onFitDataCall(self):
        self.logger.debug('onFitDataCall()')
        if self.checkDysmalPyParams(require_data=True):
            self.fitDataAsync()
    
    @pyqtSlot()
    def onLoadFittingResultCall(self):
        self.logger.debug('onLoadFittingResultCall()')
        if self.checkDysmalPyParams(require_data=True):
            self.fitDataAsync(doFit=False, clearFitResults=False)
    
    def checkDysmalPyParams(self, require_data=False, check_limits=True):
        self.logger.debug('checkDysmalPyParams()')
        checkOK = True
        errormessages = []
        if self.DysmalPyParams is None:
            errormessages.append('DysmalPyParams is None!')
            checkOK = False
        else:
            for keyname in ['z']:
                if not (keyname in self.DysmalPyParams):
                    errormessages.append('Key '+keyname+' not set!')
                    checkOK = False
                elif self.DysmalPyParams[keyname] is None or self.DysmalPyParams[keyname] == '':
                    errormessages.append('Key '+keyname+' is empty!')
                    checkOK = False
            if require_data:
                check_data = False
                for keyname in ['fdata', 'fdata_vel', 'fdata_cube']:
                    if (keyname in self.DysmalPyParams):
                        check_data = True
                if not check_data:
                    errormessages.append('No data key defined! Please check \'fdata\', \'fdata_vel\' or \'fdata_cube\' key.')
                    checkOK = False
            if check_limits:
                for keyname in self.DysmalPyParams:
                    if keyname+'_bounds' in self.DysmalPyParams:
                        this_limits = self.DysmalPyParams[keyname+'_bounds']
                        if not np.isscalar(this_limits) and len(this_limits) == 2:
                            if self.DysmalPyParams[keyname] < this_limits[0] or self.DysmalPyParams[keyname] > this_limits[1]:
                                errormessages.append('Key '+keyname+' value '+str(self.DysmalPyParams[keyname])+\
                                    ' is not in the limit '+str(this_limits)+' as defined by key '+keyname+'_bounds!')
                                checkOK = False
        if not checkOK:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("Error! DysmalPyParams is invalid!")
            msgBox.setInformativeText(self.tr('Error messages:')+'\n'+'\n'.join(errormessages))
            msgBox.setWindowTitle("Invalid DysmalPyParams")
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.exec()
        return checkOK
    
    @pyqtSlot()
    def fitDataAsync(self, doFit=True, clearFitResults=True):
        self.logger.debug('fitDataAsync(doFit={},clearFitResults={})'.format(doFit, clearFitResults))
        print('fitDataAsync(doFit={},clearFitResults={})'.format(doFit, clearFitResults))
        #
        if self.DysmalPyFittingTower.CurrentStarship is not None:
            if self.DysmalPyFittingTower.CurrentStarship.busy:
                self.logMessage(self.tr('Current subprocess is still running.'))
                return
            if clearFitResults:
                self.DysmalPyFittingTower.DysmalPyFitResults = None
        # 
        self.setButtonsEnabledDisabled(allDisabled=True)
        # pass data to DysmalPyFittingTower, which will be redirected to DysmalPyFittingStarship
        # self.DysmalPyFittingTower.BaseQueue.put( ('command', 'selectStarshipName', 'M66') )
        # self.DysmalPyFittingTower.BaseQueue.put( ('data', 'params', self.DysmalPyParams) )
        self.logger.debug('self.DysmalPyFittingTower.BaseQueue.put command fit_data')
        self.DysmalPyFittingTower.BaseQueue.put(
            (
                'command',
                'fit_data',
                ( self.DysmalPyParams, ),
                { 'do_fit': doFit,
                  'overwrite': self.DysmalPyParams['overwrite'], 
                  'param_filename': self.DysmalPyParamFile, 
                }
            )
        )
    
    @pyqtSlot()
    def generateModelCubeAsync(self, clearFitResults=False):
        self.logger.debug('generateModelCubeAsync()')
        #
        if self.DysmalPyFittingTower.CurrentStarship is not None:
            if self.DysmalPyFittingTower.CurrentStarship.busy:
                self.logMessage(self.tr('Current subprocess is still running.'))
                return
            if clearFitResults:
                self.DysmalPyFittingTower.DysmalPyFitResults = None
        #
        self.setButtonsEnabledDisabled(allDisabled=True)
        # pass data to DysmalPyFittingTower, which will be redirected to DysmalPyFittingStarship
        # self.DysmalPyFittingTower.BaseQueue.put( ('command', 'selectStarshipName', 'M66') )
        # self.logger.debug('self.DysmalPyFittingTower.BaseQueue.put data DysmalPyParams')
        # self.DysmalPyFittingTower.BaseQueue.put( ('data', 'DysmalPyParams', self.DysmalPyParams) )
        self.logger.debug('self.DysmalPyFittingTower.BaseQueue.put command fit_data')
        self.DysmalPyFittingTower.BaseQueue.put(
            (
                'command',
                'generate_model_cube',
                ( self.DysmalPyParams, ),
                { }
            )
        )
    
    @pyqtSlot()
    def generateMomentMapsAsync(self, clearFitResults=False):
        self.logger.debug('generateMomentMapsAsync()')
        #
        if self.DysmalPyFittingTower.CurrentStarship is not None:
            if self.DysmalPyFittingTower.CurrentStarship.busy:
                self.logMessage(self.tr('Current subprocess is still running.'))
                return
            if clearFitResults:
                self.DysmalPyFittingTower.DysmalPyFitResults = None
        #
        self.setButtonsEnabledDisabled(allDisabled=True)
        # pass data to DysmalPyFittingTower, which will be redirected to DysmalPyFittingStarship
        # self.DysmalPyFittingTower.BaseQueue.put( ('command', 'selectStarshipName', 'M66') )
        # self.DysmalPyFittingTower.BaseQueue.put( ('data', 'DysmalPyParams', self.DysmalPyParams) )
        self.DysmalPyFittingTower.BaseQueue.put(
            (
                'command',
                'generate_moment_maps',
                ( self.DysmalPyParams,
                  None ),
                { }
            )
        )
    
    @pyqtSlot()
    def generateRotationCurvesAsync(self, clearFitResults=False):
        self.logger.debug('generateRotationCurvesAsync()')
        #
        if self.DysmalPyFittingTower.CurrentStarship is not None:
            if self.DysmalPyFittingTower.CurrentStarship.busy:
                self.logMessage(self.tr('Current subprocess is still running.'))
                return
            if clearFitResults:
                self.DysmalPyFittingTower.DysmalPyFitResults = None
        #
        self.setButtonsEnabledDisabled(allDisabled=True)
        # pass data to DysmalPyFittingTower, which will be redirected to DysmalPyFittingStarship
        # self.DysmalPyFittingTower.BaseQueue.put( ('command', 'selectStarshipName', 'M66') )
        # self.DysmalPyFittingTower.BaseQueue.put( ('data', 'DysmalPyParams', self.DysmalPyParams) )
        self.DysmalPyFittingTower.BaseQueue.put(
            (
                'command',
                'generate_rotation_curves',
                ( self.DysmalPyParams,
                  None ),
                { }
            )
        )
    
    @pyqtSlot()
    def onFittingWorkerFinished(self):
        self.logger.debug('onFittingWorkerFinished()')
        #<TODO># we can optimize here, no need to update all these things if generateMomentMapsAsync or generateRotationCurvesAsync is run.
        self.plotDataMomentMaps()
        self.plotModelMomentMaps()
        self.plotModelRotationCurves()
        self.updateFittingParams()
        self.setButtonsEnabledDisabled()
    
    @pyqtSlot(str)
    def onFittingWorkerFinishedWithError(self, errormessage):
        self.logger.debug('onFittingWorkerFinishedWithError()')
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText("Error! Failed to run DysmalPyFittingWorker!")
        msgBox.setInformativeText(self.tr('Error messages:')+'\n'+errormessage)
        msgBox.setWindowTitle("Failed to run DysmalPyFittingWorker")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()
        self.setButtonsEnabledDisabled()
    
    @pyqtSlot(str)
    def onFittingWorkerFinishedWithWarning(self, errormessage):
        self.logger.debug('onFittingWorkerFinishedWithWarning()')
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setText("Warning from DysmalPyFittingWorker:")
        msgBox.setInformativeText(self.tr('Warning messages:')+'\n'+errormessage)
        msgBox.setWindowTitle("Warning from DysmalPyFittingWorker")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()
        self.setButtonsEnabledDisabled()
    
    def reportProgress(self):
        pass
    
    def updateFittingParams(self):
        self.logger.debug('updateFittingParams()')
        if self.DysmalPyFittingTower.DysmalPyFitResults is not None:
            # self.DysmalPyFittingTower.BaseQueue( ('command', 'sendDataToBase', 'DysmalPyFitResults') )
            free_param_names = self.DysmalPyFittingTower.DysmalPyFitResults.free_param_names
            bestfit_parameters = self.DysmalPyFittingTower.DysmalPyFitResults.bestfit_parameters
            #bestfit_parameters_l68_err = self.DysmalPyFittingTower.DysmalPyFitResults.bestfit_parameters_l68_err
            #bestfit_parameters_u68_err = self.DysmalPyFittingTower.DysmalPyFitResults.bestfit_parameters_u68_err
            self.logger.debug("updateFittingParams() DysmalPyFittingWorker.DysmalPyFitResults -> UI")
            self.logger.debug("free_param_names = %s"%(free_param_names))
            self.logger.debug("bestfit_parameters = [%s]"%(', '.join(np.array(bestfit_parameters).astype(str))))
            ipar = 0
            for component_name in free_param_names:
                for param_name in free_param_names[component_name]:
                    #self.logger.debug("bestfit %r %r = %r"%(component_name, param_name, bestfit_parameters[ipar]))
                    for i in range(len(self.LineEditModelParamsDicts)):
                        if param_name in self.LineEditModelParamsDicts[i]:
                            self.LineEditModelParamsDicts[i][param_name].setText(bestfit_parameters[ipar])
                    ipar += 1
            #for key in bestfit_parameters:
            #    self.logger.debug("bestfit_parameters[%r] = %r"%(key, bestfit_parameters[key]))
        else:
            self.logger.debug('updateFittingParams did nothing')
        return
    
    def getSlitShapeInPixel(self):
        xcenter = None
        ycenter = None
        fov_npix = None
        slit_width = None
        slit_pa = None
        pixscale = None
        slit_shape_in_pixel = None
        if self.DysmalPyParams is not None:
            if 'xcenter' in self.DysmalPyParams:
                if self.DysmalPyParams['xcenter'] is not None:
                    xcenter = self.DysmalPyParams['xcenter']
            if xcenter is None:
                if 'fov_npix' in self.DysmalPyParams:
                    if self.DysmalPyParams['fov_npix'] is not None:
                        xcenter = (self.DysmalPyParams['fov_npix']-1.0)/2.0
            #
            if 'ycenter' in self.DysmalPyParams:
                if self.DysmalPyParams['ycenter'] is not None:
                    ycenter = self.DysmalPyParams['ycenter']
            if ycenter is None:
                if 'fov_npix' in self.DysmalPyParams:
                    if self.DysmalPyParams['fov_npix'] is not None:
                        ycenter = (self.DysmalPyParams['fov_npix']-1.0)/2.0
            #
            if 'fov_npix' in self.DysmalPyParams:
                if self.DysmalPyParams['fov_npix'] is not None:
                    fov_npix = self.DysmalPyParams['fov_npix']
            #
            if 'slit_width' in self.DysmalPyParams:
                if self.DysmalPyParams['slit_width'] is not None:
                    slit_width = self.DysmalPyParams['slit_width']
            #
            if 'slit_pa' in self.DysmalPyParams:
                if self.DysmalPyParams['slit_pa'] is not None:
                    slit_pa = self.DysmalPyParams['slit_pa']
            #
            if 'pixscale' in self.DysmalPyParams:
                if self.DysmalPyParams['pixscale'] is not None:
                    pixscale = self.DysmalPyParams['pixscale']
            #
            if np.all([t is not None for t in [xcenter, ycenter, fov_npix, slit_width, slit_pa, pixscale]]):
                nx = fov_npix
                ny = fov_npix
                #if np.isclose(slit_pa, 0.0) or np.isclose(slit_pa, 180.0):
                #    x = np.array([xcenter-slit_width/pixscale/2.0, xcenter+slit_width/pixscale/2.0])
                #    y1 = np.array([0, 0]).astype(float)
                #    y2 = np.array([ny-1, ny-1]).astype(float)
                #    mask = np.array([True, True])
                #else:
                #    x = np.arange(nx).astype(float)
                #    y = (x-xcenter) * np.tan(np.deg2rad(slit_pa+90.0)) + ycenter
                #    y1 = y - np.abs(slit_width/pixscale/2.0*np.cos(np.deg2rad(slit_pa+90.0)))
                #    y2 = y + np.abs(slit_width/pixscale/2.0*np.cos(np.deg2rad(slit_pa+90.0)))
                #    mask = np.logical_and.reduce((x>=0, y1>=0, y2>=0, x<=nx-1, y1<=ny-1, y2<=ny-1))
                #slit_shape_in_pixel = {'x':x, 'y1':y1, 'y2':y2, 'where':mask, 'step':'mid', 'alpha':0.445, 'color':'cyan'}
                # slit transparency 0.445
                #self.logger.debug('slit_shape_in_pixel: '+str(slit_shape_in_pixel))
                #
                # use rectangle slit
                dx_pix = fov_npix
                dy_pix = slit_width/pixscale
                x0 = xcenter - (dx_pix/2.0*np.cos(np.deg2rad(slit_pa+90.0))) \
                             + (dy_pix/2.0*np.sin(np.deg2rad(slit_pa+90.0)))
                y0 = ycenter - (dx_pix/2.0*np.sin(np.deg2rad(slit_pa+90.0))) \
                             - (dy_pix/2.0*np.cos(np.deg2rad(slit_pa+90.0)))
                self.logger.debug('Slit x0 %s y0 %s w %s h %s angle %s'%(x0, y0, dx_pix, dy_pix, slit_pa+90.0))
                slit_shape_in_pixel = Rectangle((x0, y0), dx_pix, dy_pix, angle=slit_pa+90.0,
                                                alpha=0.225, facecolor='cyan', edgecolor='cyan')
        #
        return slit_shape_in_pixel
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        #super(QDysmalPyGUI, self).dragEnterEvent(event)
    
    def dropEvent(self, event):
        if self.ButtonOpenParamsFile.isEnabled():
            for url in event.mimeData().urls():
                filepath = url.toLocalFile()
                if filepath.endswith('.params'):
                    self.selectParamFile(filepath)
                break
            event.accept()
        #super(QDysmalPyGUI, self).dropEvent(event)



class QDysmalPyFittingTower(QObject, Thread):
    """A control tower that communicates the worker subprocess with the main GUI.
    
    It lives in its own thread, thus can not respond any pyqt signal, but it can emit pyqt signal
    to the main GUI.
    
    The main GUI needs to send a command to the queue of this class, e.g., in main GUI,
    
    ```
    tower.queue.send( ( 'message', 'hello' ) )
    tower.queue.send( ( 'data', 'flux_map', np.zeros((100,100)) ) )
    tower.queue.send( ( 'command', 'start fitting', tuple(), dict({}) ) )
    ```
    """
    
    LogMessageUpdateSignal = pyqtSignal(str)
    started = pyqtSignal()
    finished = pyqtSignal()
    finishedWithWarning = pyqtSignal(str)
    finishedWithError = pyqtSignal(str)
    
    def __init__(self, parent=None):
        #
        QObject.__init__(self, parent)
        Thread.__init__(self)
        #
        self.logger = logging.getLogger('QDysmalPyFittingTower')
        self.logger.setLevel(logging.getLogger(__name__).level)
        #
        self.logger.debug('proc id: %s, thread: %s, init'%(str(multiprocessing.current_process().pid),
                                                           str(hex(threading.currentThread().ident))))
        #
        self.MultiProcManager = multiprocessing.Manager()
        self.BaseQueue = self.MultiProcManager.Queue()
        self.logger.debug('BaseQueue ' + str(hex(id(self.BaseQueue))))
        # self.QueueForStarship = self.MultiProcManager.Queue()
        # self.logger.debug('self.QueueForStarship ' + str(hex(id(self.QueueForStarship))))
        #
        self.SharedMemoryManager = None
        # self.SharedMemoryManager = SharedMemoryManager()
        # self.SharedMemoryManager.start() # use MultiProcManager.dict instead
        #
        self.starships = OrderedDict()
        self.queues = OrderedDict()
        self.shares = OrderedDict()
        self.queues['base'] = self.BaseQueue
        self.shares['base'] = self.MultiProcManager.dict()
        self.vacancies = OrderedDict()
        self.queue_logging = None
        self.logging_listener = None
        self.queue_waiting_interval_for_base = 0.0 # taking turn to listen to base/starships with this interval in seconds
        self.queue_waiting_interval_for_starships = 0.0 # taking turn to listen to base/starships with this interval in seconds
        self.queue_max_waiting_interval_for_base = 10.0 # we can stay longer on listening to base
        self.queue_max_waiting_interval_for_starships = 3.0 # stay shorter on listening to ships
        #
        self.CurrentStarship = None
        self.CurrentStarshipId = ''
        #
        self.LastLogMessage = ''
        #
        self.DysmalPyParams = None
        self.DysmalPyGal = None
        self.DysmalPyFitDict = None
        self.DysmalPyFitResults = None
        self.DysmalPyFitResultFile = None
        #
        # self.data_cube = None
        self.data_flux_map = None
        self.data_vel_map = None
        self.data_disp_map = None
        self.data_flux_err_map = None
        self.data_vel_err_map = None
        self.data_disp_err_map = None
        self.data_mask_map = None
        self.data_flux_curve = None
        self.data_vel_curve = None
        self.data_disp_curve = None
        self.data_rotation_curve = None
        # self.model_cube = None
        self.model_cube_data_array = None
        self.model_cube_header_info = None
        self.model_flux_map = None
        self.model_vel_map = None
        self.model_disp_map = None
        self.model_flux_curve = None
        self.model_vel_curve = None
        self.model_disp_curve = None
        self.model_rotation_curve = None
        self.residual_flux_map = None
        self.residual_vel_map = None
        self.residual_disp_map = None
        self.residual_flux_curve = None
        self.residual_vel_curve = None
        self.residual_disp_curve = None
        self.residual_rotation_curve = None
        #
        self.lensing_transformer_image_plane_data_cube = None
        self.lensing_transformer_image_plane_data_info = None
        self.lensing_transformer_source_plane_data_cube = None
        self.lensing_transformer_source_plane_data_info = None
        #
        self.list_of_data_attr = [
            'DysmalPyFitResults', 'DysmalPyFitResultFile', 
            # 'data_cube', # -- can not be pickle'd thus can not be put into shared dict
            'data_flux_map', 'data_vel_map', 'data_disp_map', 'data_mask_map',
            'data_flux_curve', 'data_vel_curve', 'data_disp_curve', 'data_rotation_curve',
            # 'model_cube', # -- can not be pickle'd thus can not be put into shared dict
            'model_cube_data_array', 'model_cube_header_info', 
            'model_flux_map', 'model_vel_map', 'model_disp_map',
            'model_flux_curve', 'model_vel_curve', 'model_disp_curve', 'model_rotation_curve',
            'residual_flux_map', 'residual_vel_map', 'residual_disp_map',
            'residual_flux_curve', 'residual_vel_curve', 'residual_disp_curve', 'residual_rotation_curve',
            'lensing_transformer_image_plane_data_cube', 'lensing_transformer_image_plane_data_info',
            'lensing_transformer_source_plane_data_cube', 'lensing_transformer_source_plane_data_info',
        ]
    
    def __del__(self):
        if self.logging_listener is not None:
            self.logging_listener.stop()
        if self.SharedMemoryManager is not None:
            self.SharedMemoryManager.shutdown()
    
    def addStarship(self, this_ship, queue_in = None, queue_out = None, queue_logging = None, shared_dict = None):
        self.logger.debug('addStarship ' + str(hex(id(this_ship))))
        if queue_in is None:
            self.logger.debug('addStarship creating queue_in')
            queue_in = self.MultiProcManager.Queue() # create a communication queue
        else:
            self.logger.debug('addStarship using queue_in from input')
        self.logger.debug('StarshipQueue queue_in ' + str(hex(id(queue_in))))
        if queue_out is None:
            self.logger.debug('addStarship creating queue_out')
            queue_out = self.MultiProcManager.Queue() # create a communication queue
        else:
            self.logger.debug('addStarship using queue_out from input')
        if queue_logging is None:
            self.logger.debug('addStarship creating queue_logging and starting logging_listener')
            queue_logging = self.MultiProcManager.Queue() # create a communication queue for logging
            if self.logging_listener is not None:
                self.logging_listener.stop()
                del self.logging_listener
                self.logging_listener = None
            logging_streamhandler = logging.StreamHandler()
            # logging_streamhandler = logging.StreamHandler(sys.stdout)
            # logging_streamhandler.setLevel(self.logger.level)
            logging_streamhandler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(processName)s:%(message)s'))
            # logging_filehandler =
            self.logging_listener = QueueListener(queue_logging, logging_streamhandler, respect_handler_level = True)
            self.logging_listener.start()
        else:
            self.logger.debug('addStarship using queue_logging from input')
        if shared_dict is None:
            self.logger.debug('addStarship creating shared_dict')
            shared_dict = self.MultiProcManager.dict() # create a shared memory dict
        else:
            self.logger.debug('addStarship using shared_dict from input')
        self.logger.debug('StarshipQueue queue_out ' + str(hex(id(queue_out))))
        this_ship.connect_to_tower(self)
        this_ship.connect_to_queue(queue_in = queue_in, queue_out = queue_out, queue_logging = queue_logging)
        this_ship.connect_to_shared_memory(self.SharedMemoryManager)
        this_ship.connect_to_shared_dict(shared_dict)
        self.CurrentStarshipId = this_ship.ship_id
        self.starships[self.CurrentStarshipId] = this_ship
        self.queues[self.CurrentStarshipId] = queue_out
        self.shares[self.CurrentStarshipId] = shared_dict
        self.vacancies[self.CurrentStarshipId] = True
        self.queue_logging = queue_logging
        self.CurrentStarship = self.starships[self.CurrentStarshipId]
    
    def selectCurrentStarshipId(self, this_ship_id):
        if this_ship_id in self.starships:
            self.CurrentStarshipId = this_ship_id
            self.CurrentStarship = self.starships[self.CurrentStarshipId]
        else:
            self.CurrentStarshipId = None
            self.CurrentStarship = None
    
    def selectCurrentStarshipName(self, this_ship_name):
        self.CurrentStarshipId = None
        self.CurrentStarship = None
        for this_ship_id in self.starships:
            if this_ship_name == str(self.starships[this_ship_id].name):
                self.CurrentStarshipId = this_ship_name
                self.CurrentStarship = self.starships[self.CurrentStarshipId]
                break
    
    def logMessage(self, message):
        self.LastLogMessage = message
        self.LogMessageUpdateSignal.emit(message)
    
    def copyDataFromSharedDict(self, data_type, this_ship_id):
        if data_type in self.list_of_data_attr:
            if this_ship_id in self.starships:
                if data_type in self.shares[this_ship_id]:
                    self.logger.debug('copyDataFromSharedDict is setting ' + str(data_type))
                    setattr(self, data_type, self.shares[this_ship_id][data_type])
                else:
                    self.logger.debug('copyDataFromSharedDict is setting to None')
                    setattr(self, data_type, None)
            else:
                self.logger.debug('copyDataFromSharedDict is setting to None')
                setattr(self, data_type, None)
        else:
            self.logger.debug('copyDataFromSharedDict can not set ' + str(data_type) + 
                              'because it is not in self.list_of_data_attr')
    
    def run(self):
        #
        self.logger.debug('proc id: %s, thread: %s, run'%(str(multiprocessing.current_process().pid),
                                                          str(hex(threading.currentThread().ident))))
        #
        if len(self.starships) <= 0:
            self.logMessage(self.tr('Error! No starship connected to the control tower!'))
            return
        #
        waiting_interval = 0.0
        marked_exit = False
        while not marked_exit:
            # listening to all starships and base,
            # taking turn in a loop of 2.5 second interval
            had_news = True
            for sender_id in self.queues.keys():
                try:
                    if sender_id == 'base':
                        waiting_interval = self.queue_waiting_interval_for_base
                    else:
                        waiting_interval = self.queue_waiting_interval_for_starships
                    self.logger.debug('queues[\'' + str(sender_id) + '\'].get (timeout = ' +
                                      str(waiting_interval) + ')')
                    queue_package = self.queues[sender_id].get(True, waiting_interval)
                    # self.logger.debug('queues[\'' + str(sender_id) + '\'].get_nowait')
                    # queue_package = self.queues[sender_id].get_nowait()
                except QueueEmpty:
                    # self.logger.debug('queue is empty. continue.')
                    # had_news = False
                    # if had_news:
                    #     pass # time.sleep(0.05) # if using get_nowait, we sleep for a while here
                    # else:
                    #     self.logger.debug('queue get waiting for 2.5 secs.')
                    #     time.sleep(2.5)  # if using get_nowait, we sleep for a while here
                    #
                    # if we did not get a query package for the inquiry waiting time,
                    # the next inquiry will have a longer waiting time,
                    # but no more than 3 seconds
                    if sender_id == 'base':
                        self.queue_waiting_interval_for_base = min(
                            waiting_interval + 1.0, 
                            self.queue_max_waiting_interval_for_base
                        )
                    else:
                        self.queue_waiting_interval_for_starships = min(
                            waiting_interval + 0.5, 
                            self.queue_max_waiting_interval_for_starships
                        )
                    continue
                else:
                    # self.logger.debug('queue_package: ' + str(queue_package))
                    # if we successfully got a query package, the next inquiry will have no waiting time
                    if sender_id == 'base':
                        self.queue_waiting_interval_for_base = 0.0
                        self.queue_waiting_interval_for_starships = 0.0
                        self.logger.debug('queue_package received from base: ' + str(queue_package)[0:50])
                    else:
                        self.queue_waiting_interval_for_base = 0.0
                        self.queue_waiting_interval_for_starships = 1.0
                        self.logger.debug('queue_package received from starship ' + sender_id + ': ' + str(queue_package)[0:50])
                    if isinstance(queue_package, tuple):
                        # the first element of a queue_package should be a queue_package_type
                        # it can be 'command', 'data', 'message', or 'signal'
                        queue_package_type = 'none'
                        if isinstance(queue_package, (tuple, list)):
                            queue_package_type = queue_package[0]
                        elif isinstance(queue_package, str):
                            queue_package_type = queue_package
                        #
                        if queue_package_type == 'command':
                            # receiving a command,
                            # pass the queue_package to a starship to let it start a fitting process
                            command_name = ''
                            command_args = tuple()
                            command_kwargs = dict({ })
                            try:
                                _, command_name, command_args, command_kwargs = queue_package
                            except ValueError:
                                try:
                                    _, command_name, command_args = queue_package
                                except ValueError:
                                    try:
                                        _, command_name = queue_package
                                    except ValueError:
                                        self.logger.debug('queue_package is an invalid command, discarding it!')
                            #
                            if command_name != '':
                                if sender_id == 'base':
                                    # for the command from base, let some starship to do it
                                    # currently we use the lastly added one
                                    self.logger.debug('queue_package is a command, sending it to starship ' +
                                                      str(self.CurrentStarshipId) + ' queue ' +
                                                      str(hex(id(self.starships[self.CurrentStarshipId].queue_in))))
                                    self.starships[self.CurrentStarshipId].queue_in.put(queue_package)
                                    time.sleep(0.25)
                                else:
                                    # for the command from starships,
                                    # we process the 'copyDataFromSharedDict' command
                                    if command_name == 'copyDataFromSharedDict':
                                        # this command needs two args: data_type, this_ship_id
                                        if len(command_args) >= 1:
                                            self.logger.debug('queue_package is a command from starship, ' +
                                                              str(queue_package))
                                            for command_arg in command_args:
                                                self.copyDataFromSharedDict(command_arg, sender_id)
                                        else:
                                            self.logger.debug('queue_package is a command from starship, ' +
                                                              str(queue_package) + ', ' +
                                                              'but args number is incorrect, ' +
                                                              'discarding it!')
                                    else:
                                        # for other commands from some starship, we discard them
                                        self.logger.debug('queue_package is a command from starship, ' +
                                                          str(queue_package) + ', ' +
                                                          'discarding it.')
                                        pass
                                #
                        elif queue_package_type == 'data':
                            # receiving a data,
                            # store it in this class
                            queue_package_data_type = ''
                            queue_package_data_content = None
                            try:
                                _, queue_package_data_type, queue_package_data_content = queue_package
                            except ValueError:
                                self.logger.error('queue_package is an invalid data, discarding it!')
                            else:
                                if sender_id == 'base':
                                    # for data from base, pass them to some starship
                                    # currently we use the lastly added one
                                    self.queues[self.CurrentStarshipId].put(queue_package)
                                else:
                                    # for data from some starship, store in this class
                                    if hasattr(self, queue_package_data_type):
                                        self.logger.debug('queue_package is data, storing it into '+\
                                                          str(queue_package_data_type))
                                        setattr(self, queue_package_data_type, queue_package_data_content)
                                    else:
                                        self.logger.error('queue_package is data but not recognized '+\
                                                          str(queue_package_data_type))
                        #
                        elif queue_package_type == 'message':
                            # the child process sends a message to this class,
                            # this class sends the message to the GUI.
                            message_content = None
                            try:
                                _, message_content = queue_package
                            except ValueError:
                                self.logger.error('queue_package is an invalid message, discarding it!')
                            else:
                                self.logger.debug('queue_package is a message')
                                self.logMessage(message_content)
                        #
                        elif queue_package_type == 'signal':
                            # the child process sends a signal to this class,
                            # this class emits pyqtSignal to the main GUI.
                            signal_name = ''
                            signal_content = tuple()
                            try:
                                _, signal_name, signal_content = queue_package
                            except ValueError:
                                try:
                                    _, signal_name = queue_package
                                except ValueError:
                                    self.logger.error('queue_package is an invalid signal, discarding it!')
                            #
                            if signal_name != '':
                                self.logger.debug('queue_package is a signal')
                                if sender_id == 'base' and signal_name == 'exit':
                                    marked_exit = True
                                    for starship_id in self.starships.keys():
                                        self.starships[starship_id].queue_in.put( ('signal', 'exit') )
                                    self.logger.debug('queue break signal received. break now.')
                                    break # break out of the for loop, and ends the while loop
                                #
                                elif hasattr(self, signal_name):
                                    this_signal = getattr(self, signal_name)
                                    self.logger.debug('queue signal ' + signal_name)
                                    # self.logger.debug('this_signal ' + str(this_signal))
                                    if hasattr(this_signal, 'emit'):
                                        if signal_name == 'started':
                                            self.vacancies[sender_id] = False
                                        elif signal_name.startswith('finished'):
                                            self.vacancies[sender_id] = True
                                        self.logger.debug('queue signal ' + signal_name +
                                                          ' emitted as pyqtSignal')
                                        if len(signal_content) > 0:
                                            if isinstance(signal_content, str):
                                                this_signal.emit(signal_content)
                                            else:
                                                this_signal.emit(*signal_content)
                                        else:
                                            this_signal.emit()
                            else:
                                self.logger.error('queue_package is a signal with no name? discarding it!')
                        #
                        else:
                            self.logger.debug('queue_package is of an unknown type ' + str(queue_package))
                        #
                        # end if queue_package_type
                    #
                    else:
                        self.logger.debug('queue_package is not a tuple!')
                    #
                    # end if isinstance(queue_package, tuple)
                #
                # end if queue_package is not None
            #
            # end for queues
        #
        # end while



class QDysmalPyFittingStarship(multiprocessing.context.SpawnProcess):
# class QDysmalPyFittingStarship(Process):
    """A class to run DysmalPy data model fitting.
    
    It lives in its own subprocess, and keeps listening messages from multiprocessing.Queue.
    
    It has a child process for data model fitting, because some functions like scipy.linalg.inv
    break and cause SIGSEGV or SIGBUS error in a QThread, but not in a separate process (as tested
    by the author).
    
    This is a pure Python class, so we try to follow the PEP 8 syntax.
    """
    
    def __init__(self, tower = None, queue_in = None, queue_out = None, queue_logging = None, ship_name = None):
        #
        # super(QDysmalPyFittingStarship, self).__init__()
        multiprocessing.context.SpawnProcess.__init__(self)
        #
        import logging
        logging.basicConfig()
        self.logger = logging.getLogger('QDysmalPyFittingStarship')
        self.logger.setLevel(logging.getLogger(__name__).level)
        #
        # debug logging
        # print('QDysmalPyFittingStarship init logging.getLevelName(logging.getLogger(__name__).level) = ' + 
        #       str(logging.getLevelName(logging.getLogger(__name__).level)))
        # print('hex(id(self.logger))', hex(id(self.logger)))
        # self.logger.setLevel(logging.DEBUG)
        #
        # logging in multiprocessing has an issue
        # see https://github.com/jruere/multiprocessing-logging/blob/master/multiprocessing_logging.py
        # see https://stackoverflow.com/questions/20332359/logging-with-multiprocessing-madness
        # see https://rob-blackbourn.medium.com/how-to-use-python-logging-queuehandler-with-dictconfig-1e8b1284e27a
        self.logger_queue_handler = None
        self.logger_logging_level = self.logger.level
        if queue_logging is not None:
            #
            # print('len(self.logger.handlers) %s' % (len(self.logger.handlers)))
            # for i, orig_handler in enumerate(list(self.logger.handlers)):
            #     print('handler i %s handler %s'%(i, orig_handler))
            #     orig_handler.close()
            #     self.logger.removeHandler(orig_handler)
            #
            for i, orig_handler in enumerate(list(self.logger.handlers)):
                print('handler i %s handler %s'%(i, orig_handler))
                orig_handler.close()
                self.logger.removeHandler(orig_handler)
            #
            logger_queue_handler = QueueHandler(queue_logging)
            logger_queue_handler.setLevel(self.logger.level)
            self.logger.addHandler(logger_queue_handler)
            #
            # print('len(self.logger.handlers) %s' % (len(self.logger.handlers)))
            # for i, orig_handler in enumerate(list(self.logger.handlers)):
            #     print('handler i %s handler %s'%(i, orig_handler))
        #
        self.logger.debug('proc id: %s, thread: %s, init'%(str(multiprocessing.current_process().pid),
                                                           str(hex(threading.currentThread().ident)))) # may not work
        #
        # self.daemon = True
        #
        if ship_name is not None:
            self.ship_name = str(ship_name)
        else:
            self.ship_name = ''
        self.ship_id = str(multiprocessing.current_process().pid)
        self.tower = None # can't pickle QDysmalPyFittingTower objects
        self.queue_in = None
        self.queue_out = None
        self.queue_logging = None
        self.shared_memory = None
        self.shared_dict = None
        self.busy = False
        #
        # self.params = None
        self.param_filename = None
        self.gal = None
        self.DysmalPyFitResults = None
        self.DysmalPyFitResultFile = None
        #
        self.lensing_transformer = {'0': None} # use this to hold a pointer and pass to functions
        #
        self.do_fit = True # do a fitting or just trying to load previous best-fit results
        self.overwrite = False
        #
        self.data_cube = None
        self.data_flux_map = None
        self.data_vel_map = None
        self.data_disp_map = None
        self.data_flux_err_map = None
        self.data_vel_err_map = None
        self.data_disp_err_map = None
        self.data_mask_map = None
        self.data_flux_curve = None
        self.data_vel_curve = None
        self.data_disp_curve = None
        self.data_rotation_curve = None
        self.model_cube = None
        self.model_cube_data_array = None
        self.model_cube_header_info = None
        self.model_flux_map = None
        self.model_vel_map = None
        self.model_disp_map = None
        self.model_flux_curve = None
        self.model_vel_curve = None
        self.model_disp_curve = None
        self.model_rotation_curve = None
        self.residual_flux_map = None
        self.residual_vel_map = None
        self.residual_disp_map = None
        self.residual_flux_curve = None
        self.residual_vel_curve = None
        self.residual_disp_curve = None
        self.residual_rotation_curve = None
        #
        if queue_in is not None:
            self.connect_to_queue(queue_in = queue_in)
        if queue_out is not None:
            self.connect_to_queue(queue_out = queue_out)
        if queue_logging is not None:
            self.connect_to_queue(queue_logging = queue_logging)
        #
        if tower is not None:
            tower.addStarship(self, queue_in = queue_in, queue_out = queue_out, queue_logging = queue_logging)
    
    def __str__(self):
        this_str = 'Starship '
        if self.ship_name != '':
            this_str += 'name ' + self.ship_name + ', '
        this_str += 'proc id ' + str(self.ship_id) + ', '
        this_str += 'thread ' + str(hex(threading.currentThread().ident)) + ', '
        this_str += 'queue in ' + str(hex(id(self.queue_in))) + ', '
        this_str += 'queue out ' + str(hex(id(self.queue_out))) + ', '
        this_str = this_str.rstrip(', ')
        return this_str
    
    def connect_to_tower(self, tower):
        # self.tower = tower
        # self.tower.addStarship(self) # be careful about recursion
        pass
    
    def connect_to_queue(self, queue_in = None, queue_out = None, queue_logging = None):
        if queue_in is not None:
            self.queue_in = queue_in
        if queue_out is not None:
            self.queue_out = queue_out
        if queue_logging is not None:
            self.queue_logging = queue_logging
            #
            # print('len(self.logger.handlers) %s' % (len(self.logger.handlers)))
            # for i, orig_handler in enumerate(list(self.logger.handlers)):
            #     print('handler i %s handler %s'%(i, orig_handler))
            #
            for i, orig_handler in enumerate(list(self.logger.handlers)):
                orig_handler.close()
                self.logger.removeHandler(orig_handler)
            #
            # print('QDysmalPyFittingStarship connect_to_queue logging.getLevelName(logging.getLogger(__name__).level) = ' + 
            #       str(logging.getLevelName(logging.getLogger(__name__).level)))
            logger_queue_handler = QueueHandler(queue_logging)
            logger_queue_handler.setLevel(logging.getLogger(__name__).level)
            self.logger.setLevel(logging.getLogger(__name__).level)
            self.logger.addHandler(logger_queue_handler)
            #
            # print('len(self.logger.handlers) %s' % (len(self.logger.handlers)))
            # for i, orig_handler in enumerate(list(self.logger.handlers)):
            #     print('handler i %s handler %s'%(i, orig_handler))
    
    def connect_to_shared_memory(self, shared_memory):
        # self.shared_memory = shared_memory # I guess it can't be pickled for spawned processes
        pass
    
    def connect_to_shared_dict(self, shared_dict):
        self.shared_dict = shared_dict # use multiprocess.manager.Manager().dict()
        pass
    
    def log_message(self, message):
        self.last_log_message = message
        if self.queue_out is not None:
            self.logger.debug('queue_out.put message '+str(message)[0:50])
            self.queue_out.put( ('message', message) )
    
    def clear_lensing_transformer(self):
        if self.lensing_transformer['0'] is not None:
            del self.lensing_transformer['0']
        self.lensing_transformer['0'] = None
    
    def run(self):
        #
        # Deal with logging issue in multiprocessing with start method spawn
        # In my current environment -- python 3.7.9 multiprocessing,
        # after spawning this class in a new subprocess,
        # the self.logger states are all cleared somehow.
        # We have to do something here -- using a QueueHandler here,
        # and starting a QueueListener in the main process.
        # 
        # Try this
        #   self.logger.setLevel(logging.getLogger(__name__).level) 
        # this does not work because __name__ becomes '__mp_main__'
        # 
        self.logger.setLevel(self.logger_logging_level)
        # self.logger.debug('logging.root.manager.loggerDict = ' + str(logging.root.manager.loggerDict))
        for logger_name in logging.root.manager.loggerDict:
            if logger_name.lower().startswith('dysmalpy'):
                logging.getLogger(logger_name).setLevel(self.logger_logging_level)
        #
        # print('len(self.logger.handlers) %s' % (len(self.logger.handlers)))
        # for i, orig_handler in enumerate(list(self.logger.handlers)):
        #     print('handler i %s handler %s' % (i, orig_handler))
        #
        # Seems no need to set the QueueHandler again here after spawn.
        # It seems although self.logger.handlers is empty, 
        # the QueueHandler set in addStarship() still works. 
        # if len(self.logger.handlers) == 0:
        #     # spawn will somehow reset logger.. not sure why...
        #     # logging.basicConfig()
        #     logger_queue_handler = QueueHandler(self.queue_logging)
        #     logger_queue_handler.setLevel(self.logger_logging_level)
        #     self.logger.addHandler(logger_queue_handler)
        #     # logger_stream_handler = logging.StreamHandler()
        #     # self.logger.addHandler(logger_stream_handler)
        #     self.logger.debug('spwaned multiprocessing reset logger, we add queue handler back')
        #
        # print('len(self.logger.handlers) %s' % (len(self.logger.handlers)))
        # for i, orig_handler in enumerate(list(self.logger.handlers)):
        #     print('handler i %s handler %s' % (i, orig_handler))
        #
        # print proc id and thread info
        self.logger.debug('proc id: %s, thread: %s, run'%(str(multiprocessing.current_process().pid),
                                                          str(hex(threading.currentThread().ident))))
        # print('QDysmalPyFittingStarship proc id: %s, thread: %s, run'%(str(multiprocessing.current_process().pid),
        #                                                                str(hex(threading.currentThread().ident))))
        #
        if self.queue_out is None:
            self.logger.debug('No tower queue connected. ' +
                              'Please connect to a control tower and setup the queue_in and queue_out first.')
            return
        #
        marked_exit = False
        while not marked_exit:
            try:
                # print('QDysmalPyFittingStarship queue_in.get')
                self.logger.debug('queue_in.get')
                queue_package = self.queue_in.get()
                # queue_package = self.queue_in.get(True, 1.5)
            except QueueEmpty:
                self.logger.debug('queue_in is empty. break now')
                break
            else:
                #self.logger.debug('queue_package: '+str(queue_package))
                self.logger.debug('queue_package received: ' + str(queue_package)[0:50])
                if isinstance(queue_package, tuple):
                    # the first element of a queue_package should be a queue_package_type
                    # it can be 'command', 'data', 'message', or 'signal'
                    queue_package_type = 'none'
                    if isinstance(queue_package, (tuple, list)):
                        queue_package_type = queue_package[0]
                    elif isinstance(queue_package, str):
                        queue_package_type = queue_package
                    #
                    if queue_package_type == 'command':
                        command_name = ''
                        command_args = tuple()
                        command_kwargs = dict({})
                        try:
                            _, command_name, command_args, command_kwargs = queue_package
                        except:
                            try:
                                _, command_name, command_args = queue_package
                            except:
                                try:
                                    _, command_name = queue_package
                                except:
                                    self.logger.debug('queue_package is an invalid command, discarding it!')
                        #
                        # start a fitting process
                        if hasattr(self, command_name):
                            self.logger.debug('queue_package is a command, executing ' +
                                              str(command_name))
                            #
                            # here we call the fitting function, it will run for a while,
                            # and it should send a finished signal to the queue in the final.
                            getattr(self, command_name)(*command_args, **command_kwargs)
                        else:
                            self.logger.error('queue_package is a command but not recognized in this class ' +
                                              str(command_name))
                    #
                    elif queue_package_type == 'data':
                        # receiving a data,
                        # store it in this class
                        queue_package_data_type = ''
                        queue_package_data_content = None
                        try:
                            _, queue_package_data_type, queue_package_data_content = queue_package
                        except ValueError:
                            self.logger.error('queue_package is an invalid data, discarding it!')
                        else:
                            if hasattr(self, queue_package_data_type):
                                self.logger.debug('queue_package is data, storing it into ' +
                                                  str(queue_package_data_type))
                                setattr(self, queue_package_data_type, queue_package_data_content)
                            else:
                                self.logger.error('queue_package is data but not recognized in this class ' +
                                                  str(queue_package_data_type))
                    #
                    elif queue_package_type == 'message':
                        # the child process sent a message to this class,
                        # this class send the message to the GUI.
                        message_content = None
                        try:
                            _, message_content = queue_package
                        except ValueError:
                            self.logger.error('queue_package is an invalid message, discarding it!')
                        else:
                            self.logger.debug('queue_package is a message')
                            self.log_message(message_content)
                    #
                    elif queue_package_type == 'signal':
                        # receive some signal from a tower object.
                        signal_name = ''
                        signal_content = tuple()
                        try:
                            _, signal_name, signal_content = queue_package
                        except ValueError:
                            try:
                                _, signal_name = queue_package
                            except ValueError:
                                self.logger.error('queue_package is an invalid signal, discarding it!')
                        #
                        if signal_name != '':
                            self.logger.debug('queue_package is a signal')
                            if str(signal_name) == 'exit':
                                marked_exit = True
                                self.logger.debug('queue break signal recieved. break now.')
                                break  # break out, ends the while loop
                        else:
                            self.logger.error('queue_package is a signal with no name? discarding it!')
                    #
                    else:
                        self.logger.debug('queue_package is of an unknown type ' + str(queue_package))
                else:
                    self.logger.debug('queue_package is not a tuple!')
                # end if
            # end try
        # end while
    #
    
    def emit_started(self):
        self.busy = True
        if self.queue_out is not None:
            self.logger.debug('queue_out.put signal started')
            self.queue_out.put( ('signal', 'started') )
    
    def emit_finished(self):
        if self.queue_out is not None:
            self.logger.debug('queue_out.put signal finished')
            self.queue_out.put( ('signal', 'finished') )
        self.busy = False
    
    def emit_finished_with_error(self, err_msg):
        if self.queue_out is not None:
            self.logger.debug('queue_out.put signal finishedWithError')
            self.queue_out.put( ('signal', 'finishedWithError', err_msg) )
        self.busy = False
    
    def emit_finished_with_warning(self, err_msg):
        if self.queue_out is not None:
            self.logger.debug('queue_out.put signal finishedWithWarning')
            self.queue_out.put( ('signal', 'finishedWithWarning', err_msg) )
        self.busy = False
    
    def send_data_to_queue(self, list_of_data_name, list_of_data_content = None):
        """Send a list of data to queue_out. 
        
        If self.shared_dict exists, which is a multiprocessing.manager.Manager().dict(), 
        we copy data from self. variables into self.shared_dict, then send a command 
        to queue_out to let the receiver know that it is the time to copy data out of 
        self.shared_dict from the receiver side. 
        
        If self.shared_dict does not exist, that is, there is no shared memory stuff, 
        then we send the whole data content via queue_out, which is a 
        multiprocessing.manager.Queue() tunnel. This is slower than the former. 
        
        We can pass a `list_of_data_content` list of whole data content together with 
        the name list `list_of_data_name`. When it is `None`, we assume that self 
        has a variable with this data name, and get the data content via
        `getattr(self, data_name)`. 
        """
        if self.queue_out is None:
            return
        if self.shared_dict is not None:
            for i, data_name in enumerate(list_of_data_name):
                #print('DEBUG', 'data_name', data_name)
                if list_of_data_content is None:
                    self.shared_dict[data_name] = getattr(self, data_name)
                else:
                    self.shared_dict[data_name] = list_of_data_content[i]
            self.logger.debug('queue_out.put command copyDataFromSharedDict ' +
                              ' '.join(list_of_data_name))
            self.queue_out.put(
                ('command',
                 'copyDataFromSharedDict',
                 tuple(list_of_data_name),
                 dict({})
                )
            )
        else:
            for i, data_name in enumerate(list_of_data_name):
                if list_of_data_content is None:
                    data_content = getattr(self, data_name)
                else:
                    data_content = list_of_data_content[i]
                self.logger.debug('queue_out.put data ' +
                                  data_name)
                self.queue_out.put( ('data', data_name, data_content) )
    
    def send_lensing_data_to_queue(self, send_null_data = False):
        """Try to send the lensing image plane source plane data to queue. 
        
        If there is no lensing transformer, of course we do nothing. 
        """
        has_lensing_data = False
        if self.lensing_transformer['0'] is not None:
            self.lensing_transformer_source_plane_data_cube = self.lensing_transformer['0'].source_plane_data_cube
            self.lensing_transformer_source_plane_data_info = self.lensing_transformer['0'].source_plane_data_info
            self.lensing_transformer_image_plane_data_cube = self.lensing_transformer['0'].image_plane_data_cube
            self.lensing_transformer_image_plane_data_info = self.lensing_transformer['0'].image_plane_data_info
            has_lensing_data = True
        else:
            self.lensing_transformer_source_plane_data_cube = None
            self.lensing_transformer_source_plane_data_info = None
            self.lensing_transformer_image_plane_data_cube = None
            self.lensing_transformer_image_plane_data_info = None
        # 
        if has_lensing_data or send_null_data:
            self.send_data_to_queue(
                    ['lensing_transformer_source_plane_data_cube',
                     'lensing_transformer_source_plane_data_info',
                     'lensing_transformer_image_plane_data_cube',
                     'lensing_transformer_image_plane_data_info']
                )
    
    def get_dysmalpy_gal(self, params, block_signal = False):
        """ Function to call dysmalpy functions and prepare a galaxy object.
        """
        if not block_signal:
            self.emit_started()
        self.log_message('Getting dysmalpy galaxy object and fit dict from params.')
        if params is None:
            err_msg = 'Error! dysmalpy params is invalid!'
            self.log_message(err_msg)
            if not block_signal:
                self.emit_finished_with_error(err_msg)
            return
        # 
        if 'nObs' not in params:
            params['nObs'] = 1
        if 'obs_1_name' not in params:
            params['obs_1_name'] = 'obs'
        if 'obs_1_tracer' not in params:
            params['obs_1_tracer'] = 'gas'
        #
        gal = None
        fit_dict = None # 20250827 new dysmalpy uses output_options instead
        output_options = None # 20250827 new dysmalpy uses output_options instead
        #if 'fdata' in params.keys():
        #    ndim_fit = 1
        #    gal, fit_dict = setup_single_galaxy(params=params, data=None)
        #    gal.instrument = gal.observations[list(gal.observations.keys())[0]].instrument
        #elif 'fdata_vel' in params.keys():
        #    ndim_fit = 2
        #    gal, fit_dict = setup_single_galaxy(params=params, data=None)
        #    gal.instrument = gal.observations[list(gal.observations.keys())[0]].instrument
        #elif 'fdata_cube' in params.keys():
        #    ndim_fit = 3
        #    gal, fit_dict = setup_single_galaxy(params=params, data=None)
        #    gal.instrument = gal.observations[list(gal.observations.keys())[0]].instrument
        # 20240830
        if 'fdata' in params.keys() or 'fdata_vel' in params.keys() or 'fdata_cube' in params.keys(): # 20240906
            # 20250827
            #gal, fit_dict_options = setup_single_galaxy(params=params)
            #fit_dict = fit_dict_options.as_dict()
            #print('fit_dict', fit_dict) #<DZLIU><20240906># how to get fit_dict now?
            gal, output_options = setup_single_galaxy(params=params)
            fit_dict = output_options.as_dict()
            obs = gal.observations[list(gal.observations.keys())[0]]
            gal.instrument = obs.instrument
            gal.data = obs.data
            #print('gal.model.mass_components', gal.model.mass_components)
        else:
            self.log_message('Warning! No data defined in DysmalPyParams. '+
                             'Please check the \'fdata\', \'fdata_vel\' or \'fdata_cube\' keys.')
            self.log_message('Setting up galaxy model base...')
            has_psf = False
            if 'psf_fwhm' in params:
                if params['psf_fwhm'] not in [None, '', 'None']:
                    has_psf = True
            if 'psf_fwhm_major' in params:
                if params['psf_fwhm_major'] not in [None, '', 'None']:
                    has_psf = True
            if not has_psf:
                # allow user to set psf_fwhm to None, and here we do some tricks to skip the convolution with a psf
                copy_psf_fwhm = params['psf_fwhm']
                params['psf_fwhm'] = 0.1
                gal = setup_gal_model_base(params=params)
                # 20240830
                obs = make_empty_observation(0, ndim=2, params=params)
                gal.observations[params['obs_1_name']] = obs
                gal.instrument = obs.instrument
                gal.instrument.beam = None
                gal.instrument._beam_kernel = None
                params['psf_fwhm'] = copy_psf_fwhm
            else:
                gal = setup_gal_model_base(params=params)
                # 20240830
                obs = make_empty_observation(0, ndim=2, params=params)
                gal.observations[params['obs_1_name']] = obs
                gal.instrument = obs.instrument
            #gal.data = data_io.load_single_object_1D_data(fdata=params['fdata'], fdata_mask=fdata_mask, params=params, datadir=datadir)
            #gal.data = data_classes.Data1D(...)
            #gal.data.filename_velocity = datadir+params['fdata']
            #if (params['profile1d_type'] != 'circ_ap_pv') & (params['profile1d_type'] != 'single_pix_pv'):
            #gal.data.apertures = setup_basic_aperture_types(gal=gal, params=params)
            #gal.data.profile1d_type = params['profile1d_type']
            #fit_dict = setup_fit_dict(params=params, ndim_data=1)
        #
        #20230912
        #print('DEBUGGING: get_dysmalpy_gal() gal.instrument.beam: {} {} {}'.format(
        #    gal.instrument.beam._major, gal.instrument.beam._minor, gal.instrument.beam._pa))
        # 
        #return gal, fit_dict # 20250827 new dysmalpy uses output_options instead
        return gal, fit_dict, output_options # 20250827 new dysmalpy uses output_options instead
    
    def fit_data(self, params, do_fit = True, overwrite = False, param_filename = None, block_signal = False):
        #
        if not block_signal:
            self.emit_started()
        #
        if do_fit:
            self.log_message('Preparing to fit the data...')
        else:
            self.log_message('Preparing to load fitting result...')
        if params is None:
            err_msg = 'Error! dysmalpy params is invalid! Could not proceed to fit the data!'
            self.log_message(err_msg)
            if not block_signal:
                self.emit_finished_with_error(err_msg)
            return
        #
        #gal, fit_dict = self.get_dysmalpy_gal(params) # 20250827 new dysmalpy uses output_options instead
        gal, fit_dict, output_options = self.get_dysmalpy_gal(params) # 20250827 new dysmalpy uses output_options instead
        #
        if gal is None or fit_dict is None:
            err_msg = 'Error! Could not get DysmalPyGal and DysmalPyFitDict from DysmalPyParams?!'
            self.log_message(err_msg)
            if not block_signal:
                self.emit_finished_with_error(err_msg)
            return
        #
        check_fixed_pars = []
        check_tied_pars = []
        check_free_pars = []
        check_str = ''
        check_free_par_counter = 0
        self.logger.debug('gal.model.param_names ' + str(gal.model.param_names) + 
                          ' (' + str(len(gal.model.param_names)) + ')')
        self.logger.debug('gal.model.nparams_free ' + str(gal.model.nparams_free))
        self.logger.debug('gal.model.fixed ' + str(gal.model.fixed))
        self.logger.debug('gal.model.tied ' + str(gal.model.tied))
        for check_key in gal.model.fixed:
            check_str += '  %-13s'%(check_key)
            check_first_line = True
            for check_subkey in gal.model.tied[check_key]:
                if check_first_line:
                    check_first_line = False
                else:
                    check_str += '  %-13s'%(' ')
                # 
                if gal.model.fixed[check_key][check_subkey] != False:
                    check_str += ' %-13s'%(check_subkey) + ' fixed'
                elif gal.model.tied[check_key][check_subkey] != False:
                    check_str += ' %-13s'%(check_subkey) + ' tied'
                else:
                    check_free_par_counter += 1
                    check_str += ' %-13s'%(check_subkey) + ' free (' + str(check_free_par_counter) + ')'
                check_str += '\n'
        self.logger.debug('gal.model pars: \n' + check_str)
        # 
        # 20250827
        fitter = utils_io.setup_fitter(params=params)
        output_options.set_output_options(gal, fitter)
        output_options.overwrite = True
        # 
        # 20250827
        #fit_method = fit_dict['fit_method']
        #do_plotting = fit_dict['do_plotting']
        fit_method = fitter.fit_method.lower()
        do_plotting = output_options.do_plotting
        # 
        if fit_method == 'mcmc':
            output_pickle_file = os.path.join(params['outdir'], params['galID']+'_mcmc_results.pickle')
        elif fit_method == 'mpfit':
            output_pickle_file = os.path.join(params['outdir'], params['galID']+'_mpfit_results.pickle')
        else:
            err_msg = 'Error! fit_method must be \'mcmc\' or \'mpfit\'! Could not do the fitting!'
            self.log_message(err_msg)
            if not block_signal:
                self.emit_finished_with_error(err_msg)
            return
        self.log_message('Setting fit_method to '+fit_method+'.')
        #
        # 20250827
        #config_c_m_data = config.Config_create_model_data(**fit_dict)
        #config_sim_cube = config.Config_simulate_cube(**fit_dict)
        #kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}
        #kwargs_galmodel['lensing_transformer'] = self.lensing_transformer
        #kwargs_all = {**kwargs_galmodel, **fit_dict}
        #
        fit_results = None
        #
        if ((not os.path.isfile(output_pickle_file)) or overwrite) and do_fit:
            # check DysmalPy param limits
            #self.checkDysmalPyParams()
            # log message
            self.log_message('Fitting with fit_method ' + fit_method + '...')
            # run the DysmalPy fitting
            if fit_method == 'mcmc':
                #fit_results = fitting.fit_mcmc(gal, **kwargs_all) # 20250827 new dysmalpy
                fit_results = fitter.fit(gal, output_options)
                self.log_message('fit_mcmc done')
            elif fit_method == 'mpfit':
                #fit_results = fitting.fit_mpfit(gal, **kwargs_all) # 20250827 new dysmalpy
                fit_results = fitter.fit(gal, output_options)
                self.log_message('fit_mpfit done')
            else:
                err_msg = 'Error! fit_method must be \'mcmc\' or \'mpfit\'! Could not do the fitting!'
                self.log_message(err_msg)
                if not block_signal:
                    self.emit_finished_with_error(err_msg)
                return
            # check output file
            if not os.path.isfile(output_pickle_file):
                err_msg = 'Error! Output file not produced?! '+str(output_pickle_file)
                self.log_message(err_msg)
                if not block_signal:
                    self.emit_finished_with_error(err_msg)
                return
            else:
                self.log_message('Output to '+str(output_pickle_file)+'.')
            # 
            # Save text results
            # see dysmalpy_fit_single.py
            # 20250827 new dysmalpy has no save_results_ascii_files
            #utils_io.save_results_ascii_files(
            #    fit_results=fit_results, 
            #    gal=gal, 
            #    params=params,
            #    overwrite=overwrite)
            # 
            # Save more results
            # see XXX.py
            # 20250827 new dysmalpy has no save_results_ascii_files
            #fit_results.results_report(
            #    gal=gal,
            #    params=params,
            #    overwrite=overwrite,
            #    filename=os.path.join(params['outdir'], 
            #        str(params['galID'])+'_'+str(params['fit_method'])+'_fit_report.txt'), 
            #    )
            fit_results.results_report(
                gal=gal,
                filename=os.path.join(params['outdir'], 
                    str(params['galID'])+'_'+str(params['fit_method'])+'_fit_report.txt'), 
                output_options=output_options,
                )
            # 
            # Save more results
            # see XXX.py
            if True:
                kwargs_contour_data = {
                    'lw_cont': 0.75,
                    'delta_cont_v': 100.,
                    'delta_cont_disp': 25.,
                    'delta_cont_flux': 50.,
                    'delta_cont_v_minor': 10.,
                    'max_residual': 100.,
                    }
                # 20250827
                kwargs_contour_data = {}
                # 
                plotting.plot_data_model_comparison(
                    gal=gal,
                    show_contours=True, 
                    #fitflux=False, #<TODO>#
                    fileout=os.path.join(params['outdir'], 
                        str(params['galID'])+'_'+str(params['fit_method'])+'_fit_datmod_comp.pdf'),
                    **kwargs_contour_data
                )
            # 
            # Save params file
            # see dysmalpy_fit_single.py
            # param_filename = os.path.join(params['outdir'], 'fit.params') #<TODO># always save as this file name
            if param_filename is not None:
                #data_io.ensure_dir(params['outdir']) # 20250827?
                #fitting.ensure_dir(params['outdir'])
                if params['outdir'] is not None and params['outdir'] != '' and not os.path.exists(params['outdir']):
                    os.makedirs(params['outdir'])
                utils_io.preserve_param_file(param_filename, params=params, 
                                             datadir=params['datadir'], 
                                             outdir=params['outdir'])
                # 
                # Make component plot
                # see dysmalpy_fit_single.py
                if do_plotting:
                    ndim = gal.data.ndim
                    # 20250827
                    # if ndim == 1:
                    #     plot_bundle_1D(
                    #         params=params, fit_dict=fit_dict, param_filename=param_filename,
                    #         plot_type='pdf', overwrite=overwrite,
                    #         **kwargs_galmodel)
                    # elif ndim == 2:
                    #     plot_bundle_2D(
                    #         params=params, param_filename=param_filename, 
                    #         plot_type='pdf', overwrite=overwrite)
                    # elif ndim == 3:
                    #     pass
                    plot_1D_rotcurve_components(gal, output_options) # 20250827 new dysmalpy
                    # 
                    self.log_message('Successfully fitted the data and plotted the results.')
            else:
                # 
                self.log_message('Successfully fitted the data.')
            # 
        elif os.path.isfile(output_pickle_file):
            # load existing fitting result
            self.log_message('Loading previous fitting...')
            gal, fit_results = fitting.reload_all_fitting(\
                                        filename_galmodel=fit_dict['f_model'],
                                        filename_results=fit_dict['f_results'],
                                        fit_method=fit_method)
            self.log_message('Successfully loaded previous fitting.')
        else:
            err_msg = 'Warning! Previous fitting result not found. Please click the "Fit Data" button to run a fit.'
            self.log_message(err_msg)
            if not block_signal:
                self.emit_finished_with_warning(err_msg)
            return
        # 
        gal.data = gal.observations['OBS'].data # 20250827 fix for compatible issue
        #gal.model_cube = gal.observations['OBS'].model_cube # 20250827 fix for compatible issue
        gal.model_cube = gal.observations['OBS'].model_data # 20251117 the flux rescaled model cube is obs.model_data
        # 
        self.data_cube = None
        if hasattr(gal, 'data'):
            if hasattr(gal.data, 'data'):
                if gal.data.ndim == 3:
                    self.data_cube = copy.copy(gal.data.data)
        self.model_cube = gal.model_cube.data # 20250827 new dysmalpy
        self.model_cube_data_array = self.model_cube._data
        self.model_cube_header_info = self.model_cube._header
        # self.params = params
        self.gal = gal
        self.DysmalPyFitResults = fit_results
        self.DysmalPyFitResultFile = output_pickle_file
        # 
        self.send_data_to_queue(
                # ['data_cube', 'model_cube', 'DysmalPyFitResults', 'DysmalPyFitResultFile']
                # ['model_cube', 'DysmalPyFitResults', 'DysmalPyFitResultFile'] # tower does not need data cube
                ['model_cube_data_array', 'model_cube_header_info', 
                 'DysmalPyFitResults', 'DysmalPyFitResultFile'] # tower does not need data cube
            )
        # 
        self.send_lensing_data_to_queue()
        #
        #print('generate_moment_maps', 'L4491')
        self.generate_moment_maps(params, gal, block_signal = True)
        if self.last_log_message.startswith('Error!'):
            if not block_signal:
                self.emit_finished_with_error(self.last_log_message)
        #
        self.generate_rotation_curves(params, gal, block_signal = True)
        if self.last_log_message.startswith('Error!'):
            if not block_signal:
                self.emit_finished_with_error(self.last_log_message)
        #
        if do_fit:
            self.log_message('Successfully generated the model cube, moment maps and rotation curves.')
        #
        if not block_signal:
            self.emit_finished()
    
    def generate_model_cube(self, params, block_signal = False):
        print('generate_model_cube')
        #
        if not block_signal:
            self.emit_started()
        #
        self.log_message('Generating model cube...')
        if params is None:
            err_msg = 'Error! DysmalPyParams is invalid!' + \
                      'Could not proceed to generate the model cube.'
            self.log_message(err_msg)
            if not block_signal:
                self.emit_finished_with_error(err_msg)
            return
        #
        try:
            # 20250827
            #gal, fit_dict = self.get_dysmalpy_gal(params)
            gal, fit_dict, output_options = self.get_dysmalpy_gal(params)
        except Exception as err:
            #traceback.print_tb(err.__traceback__)
            traceback.print_exc()
            self.logger.exception(err)
            gal = None
        #
        if gal is None:
            err_msg = 'Error! Could not set DysmalPyGal from DysmalPyParams?!'
            self.log_message(err_msg)
            if not block_signal:
                self.emit_finished_with_error(err_msg)
            return
        if gal.instrument is None:
            err_msg = 'Error! DysmalPyGal.instrument is invalid! ' + \
                      'Could not proceed to generate the model cube.'
            self.log_message(err_msg)
            if not block_signal:
                self.emit_finished_with_error(err_msg)
            return
        #
        # create model cube from_instrument:
        # nx_sky, ny_sky, spec_type, spec_start, spec_step, spec_unit, nspec, rstep
        # will be read from gal.instrument
        # xcenter ycenter are needed only if the center is not at the cube center.
        # line_center is needed only if spec_type is 'wavelength'
        ndim_final = 3
        if 'profile1d_type' in params:
            profile1d_type = params['profile1d_type']
        else:
            profile1d_type = 'circ_ap_pv'
        if 'aperture_radius' in params:
            aperture_radius = params['aperture_radius'] # if profile1d_type == 'circ_ap_cube'
        else:
            aperture_radius = 0.1
        oversample = params['oversample']
        if oversample is None or str(oversample) == 'None':
            oversample = 1
            params['oversample'] = 1
        oversize = params['oversize']
        if oversize is None or str(oversize) == 'None':
            oversize = 1
            params['oversize'] = 1
        pixscale = params['pixscale']
        fov_npix = params['fov_npix']
        xcenter = params['xcenter']
        ycenter = params['ycenter']
        #aper_centers = np.linspace(-np.abs(fov_npix/pixscale/2.0), +np.abs(fov_npix/pixscale/2.0), num=30, endpoint=True) #<TODO>#
        aper_centers = np.linspace(-np.abs(fov_npix*pixscale/2.0), +np.abs(fov_npix*pixscale/2.0), num=30, endpoint=True) # in arcsec
        #kwargs_galmodel = {}
        #if profile1d_type == 'circ_ap_cube':
        #    kwargs_galmodel['aperture_radius'] = 0.1
        #elif profile1d_type == 'rect_ap_cube':
        #    kwargs_galmodel['pix_perp'] = 5
        #    kwargs_galmodel['pix_parallel'] = 20
        #elif profile1d_type == 'square_ap_cube':
        #    kwargs_galmodel['pix_length'] = 20
        kwargs_galmodel = setup_lensing_dict(params)
        kwargs_galmodel['lensing_transformer'] = self.lensing_transformer
        #kwargs_galmodel['gauss_extract_with_c'] = params['gauss_extract_with_c'] #20211111
        for key in ['moment_calc', 'gauss_extract_with_c', 'zcalc_truncate', 'zcalc_with_c']:
            if key in params:
                kwargs_galmodel[key] = params[key] #20211111
                # print(f'kwargs_galmodel[{key!r}] = params[{key!r}] = {params[key]!r}') #20211111 DEBUG
        from_data = False # must set from_data = False to let the input ndim_final = 3 in effect
        # self.logger.debug("self.lensing_transformer " + str(self.lensing_transformer))
        # self.logger.debug("kwargs_galmodel['lensing_transformer'] " + str(kwargs_galmodel['lensing_transformer']))
        # self.logger.debug("hasattr(gal, 'instrument') " + str(hasattr(gal, 'instrument')))
        # self.logger.debug("hasattr(gal, 'model') " + str(hasattr(gal, 'model')))
        gal.model._update_tied_parameters()
        # 20240830
        #obs = gal.observations[list(gal.observations.keys())[0]]
        # print('gal.create_model_data')
        # print('gal.dscale', gal.dscale)
        # print('obs.instrument.moment', obs.instrument.moment, "kwargs_galmodel['moment_calc']", kwargs_galmodel['moment_calc'])
        # print('obs.mod_options.gauss_extract_with_c', obs.mod_options.gauss_extract_with_c, "kwargs_galmodel['gauss_extract_with_c']", kwargs_galmodel['gauss_extract_with_c'])
        # print('obs.instrument.ndim', obs.instrument.ndim)
        # print('obs.instrument.fov[0]', obs.instrument.fov[0])
        # print('obs.instrument.fov[1]', obs.instrument.fov[1])
        # print('obs.instrument.spec_type', obs.instrument.spec_type)
        # print('obs.instrument.spec_step.unit', obs.instrument.spec_step.unit)
        # print('obs.instrument.nspec', obs.instrument.nspec)
        # print('obs.instrument.pixscale.value', obs.instrument.pixscale.value)
        # print('obs.mod_options.oversample', obs.mod_options.oversample)
        # print('obs.mod_options.oversize', obs.mod_options.oversize)
        # print('obs.lensing_options', obs.lensing_options)
        # print('kwargs_galmodel', kwargs_galmodel)
        if 'lensing_mesh' in params:
            obs = gal.observations[list(gal.observations.keys())[0]]
            obs.lensing_options.load(**kwargs_galmodel)
            print('obs.lensing_options.get_lensing_kwargs()', obs.lensing_options.get_lensing_kwargs(oversample=obs.mod_options.oversample, oversize=obs.mod_options.oversize))
        #print('obs.instrument.smoothing_type', obs.instrument.smoothing_type)
        #obs.create_single_obs_model_data(gal.model, gal.dscale)
        gal.create_model_data()
        obs = gal.observations[list(gal.observations.keys())[0]]
        #gal.create_model_data(\
        #                    ndim_final = ndim_final,
        #                    profile1d_type = profile1d_type,
        #                    aperture_radius = aperture_radius,
        #                    aper_centers = aper_centers,
        #                    from_instrument = True,
        #                    from_data = from_data,
        #                    oversample = oversample,
        #                    oversize = oversize,
        #                    xcenter = xcenter,
        #                    ycenter = ycenter,
        #                    **kwargs_galmodel,
        #                    )
        #print('obs.model_cube.data', obs.model_cube.data)
        #fits.PrimaryHDU(data=obs.model_cube.data).writeto('tmp.obs.model_cube.fits', overwrite=True)
        #gal.model_cube = obs.model_cube
        gal.model_cube = obs.model_data
        gal.data = obs.data
        #self.logger.debug("self.lensing_transformer " + str(self.lensing_transformer))
        #self.logger.debug("kwargs_galmodel['lensing_transformer'] " + str(kwargs_galmodel['lensing_transformer']))
        #
        # 20240830 new dysmalpy
        #self.model_cube = copy.copy(gal.model_cube.data)
        #self.model_cube_data_array = self.model_cube._data
        #self.model_cube_header_info = self.model_cube._header
        #
        # 20251117 new dysmalpy -- the flux rescaled model cube is obs.model_data
        self.model_cube = copy.copy(obs.model_data.data)
        # 
        if gal.data.mask is not None: #20251008
            self.model_cube = self.model_cube.with_mask(gal.data.mask) #20251008
            #print('self.model_cube = self.model_cube.with_mask(gal.data.mask) #20251008')
        self.model_cube_data_array = self.model_cube.filled(np.nan).value
        self.model_cube_header_info = self.model_cube._header
        # self.params = params
        self.gal = gal
        # 
        self.send_data_to_queue(
                # ['model_cube'] # can not pickle a SpectralCube
                ['model_cube_data_array', 'model_cube_header_info']
            )
        # 
        self.send_lensing_data_to_queue()
        #
        self.generate_moment_maps(params, gal, block_signal = True)
        if self.last_log_message.startswith('Error!'):
            if not block_signal:
                self.emit_finished_with_error(self.last_log_message)
            return
        #
        self.generate_rotation_curves(params, gal, block_signal = True)
        if self.last_log_message.startswith('Error!'):
            if not block_signal:
                self.emit_finished_with_error(self.last_log_message)
            return
        #
        self.logger.debug('generateModelCube: done')
        self.log_message('Successfully generated the model cube, moment maps and rotation curves.')
        if not block_signal:
            self.emit_finished()
    
    def compute_moment_maps_from_cube(self, params, data_cube, data_mask = None):
        # see dysmalpy "galaxy.py"
        if not isinstance(data_cube, SpectralCube):
            self.log_message('Error! The input data cube to compute_moment_maps_from_cube is not a SpectralCube!')
            return None, None, None

        if data_mask is not None:
            data_cube = data_cube.with_mask(data_mask)
            #print('data_cube = data_cube.with_mask(data_mask) *** DEBUG ***')
        
        mom0 = data_cube.moment0().to(u.km/u.s).value
        mom1 = data_cube.moment1().to(u.km/u.s).value
        #mom2 = data_cube.linewidth_sigma().to(u.km/u.s).value # this will contain nan if mom2 has negative values
        mom2 = np.sqrt(np.abs(data_cube.moment2().to((u.km/u.s)**2).value))
        flux = np.zeros(mom0.shape)
        vel = np.zeros(mom0.shape)
        disp = np.zeros(mom0.shape)

        #dataarr = data_cube.unmasked_data[:,:,:].value
        dataarr = data_cube.filled(np.nan).value
        specarr = data_cube.spectral_axis.to(u.km/u.s).value
        specarrmin = specarr.min()
        specarrmax = specarr.max()
        mom1[mom1<specarrmin] = specarrmin+0.01*(specarrmax-specarrmin)
        mom1[mom1>specarrmax] = specarrmax-0.01*(specarrmax-specarrmin)
        
        # <DZLIU><20210805> ++++++++++
        #logger.debug('data_cube.spectral_axis.to(u.km/u.s).value: '+str(data_cube.spectral_axis.to(u.km/u.s).value))
        my_least_chi_squares_1d_fitter = None
        if 'gauss_extract_with_c' in params:
            #logger.debug('params[\'gauss_extract_with_c\'] = ' +str(params['gauss_extract_with_c']))
            if params['gauss_extract_with_c'] is not None and \
               params['gauss_extract_with_c'] is not False:
                this_fitting_mask = 'auto'
                if data_mask is not None:
                    if hasattr(data_mask, 'shape'):
                        if len(data_mask.shape) == 2:
                            this_fitting_mask = copy.copy(data_mask)
                        elif len(data_mask.shape) == 3:
                            # the C++ code has a problem that it skips a pixel if any channel is masked out, so we only input a 2D mask
                            #this_fitting_mask = (np.sum(data_mask.astype(int), axis=0)>0)
                            this_fitting_mask = copy.copy(data_mask)
                if logger.level > logging.DEBUG:
                    this_fitting_verbose = True
                else:
                    this_fitting_verbose = False
                init_amp = np.abs(mom0/np.sqrt(2*np.pi)/np.abs(mom2))
                init_mean = mom1
                init_sigma = np.abs(mom2)
                # print('***DEBUG*** np.count_nonzero(np.isnan(init_amp))', np.count_nonzero(np.isnan(init_amp)))
                # print('***DEBUG*** np.count_nonzero(np.isnan(init_mean))', np.count_nonzero(np.isnan(init_mean)))
                # print('***DEBUG*** np.count_nonzero(np.isnan(init_sigma))', np.count_nonzero(np.isnan(init_sigma)))
                # print('***DEBUG*** np.count_nonzero(np.invert(this_fitting_mask))', np.count_nonzero(np.invert(this_fitting_mask)))
                nanmax = np.nanmax(data_cube.unmasked_data[:,:,:].value)
                init_amp[np.isnan(init_amp)] = 1.0
                init_amp[init_amp > nanmax] = nanmax
                init_amp[init_amp < 0.0] = nanmax * 0.001
                init_mean[np.isnan(init_mean)] = 0.0
                init_sigma[np.isnan(init_sigma)] = 1.0
                init_sigma[init_sigma <= 1.0] = 1.0
                # 20250827 debugging
                #fits.PrimaryHDU(dataarr).writeto('tmp.dataarr.fits', overwrite=True)
                #fits.PrimaryHDU(init_amp).writeto('tmp.init_amp.fits', overwrite=True)
                #fits.PrimaryHDU(init_mean).writeto('tmp.init_mean.fits', overwrite=True)
                #fits.PrimaryHDU(init_sigma).writeto('tmp.init_sigma.fits', overwrite=True)
                my_least_chi_squares_1d_fitter = LeastChiSquares1D(\
                        x = specarr,
                        data = data_cube.unmasked_data[:,:,:].value,
                        dataerr = None,
                        datamask = this_fitting_mask,
                        initparams = np.array([init_amp, init_mean, init_sigma]),
                        nthread = 4,
                        verbose = this_fitting_verbose,
                        #nthread = 1,
                        #verbose = True,
                        #c_verbose = 3,
                    )
        if my_least_chi_squares_1d_fitter is not None:
            self.logger.debug('my_least_chi_squares_1d_fitter '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#
            try:
                my_least_chi_squares_1d_fitter.runFitting()
                flux = my_least_chi_squares_1d_fitter.outparams[0,:,:] * np.sqrt(2 * np.pi) * my_least_chi_squares_1d_fitter.outparams[2,:,:]
                vel = my_least_chi_squares_1d_fitter.outparams[1,:,:]
                disp = my_least_chi_squares_1d_fitter.outparams[2,:,:]
                badmask = np.logical_or.reduce((np.isnan(flux), flux<-1e10, flux>1e10, vel<specarrmin, vel>specarrmax))
                flux[badmask] = np.nan #<DZLIU><DEBUG># 20210809 fixing this bug
                vel[badmask] = np.nan
                disp[badmask] = np.nan
                self.logger.debug('my_least_chi_squares_1d_fitter '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#
            except:
                my_least_chi_squares_1d_fitter = None
        if my_least_chi_squares_1d_fitter is None:
            for i in range(mom0.shape[0]):
                for j in range(mom0.shape[1]):
                    if i==0 and j==0:
                        self.logger.debug('gaus_fit_sp_opt_leastsq '+str(mom0.shape[0])+'x'+str(mom0.shape[1])+' '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#
                    best_fit = gaus_fit_sp_opt_leastsq(data_cube.spectral_axis.to(u.km/u.s).value,
                                        data_cube.unmasked_data[:,i,j].value,
                                        mom0[i,j], mom1[i,j], mom2[i,j])
                    flux[i,j] = best_fit[0] * np.sqrt(2 * np.pi) * best_fit[2]
                    vel[i,j] = best_fit[1]
                    disp[i,j] = best_fit[2]
                    if i==mom0.shape[0]-1 and j==mom0.shape[1]-1:
                        self.logger.debug('gaus_fit_sp_opt_leastsq '+str(mom0.shape[0])+'x'+str(mom0.shape[1])+' '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#
        # <DZLIU><20210805> ----------

        if data_mask is not None:
            if len(data_mask.shape) == 3:
                data_mask_2D = (np.sum(data_mask.astype(int), axis=0) > 0)
                flux[~data_mask_2D] = np.nan
                vel[~data_mask_2D] = np.nan
                disp[~data_mask_2D] = np.nan

        return flux, vel, disp
    
    def generate_moment_maps(self, params, gal = None, block_signal = False):
        if not block_signal:
            self.emit_started()
        #
        self.log_message('Generating moment maps...')
        #
        if params is None:
            err_msg = 'Error! DysmalPyParams is invalid! ' + \
                      'Could not proceed to generate the moment maps. ' + \
                      'Please set dysmalpy params first!'
            self.log_message(err_msg)
            if not block_signal:
                self.emit_finished_with_error(err_msg)
            return
        if gal is None:
            gal = self.gal
        if gal is None:
            err_msg = 'Error! DysmalPyGal is invalid! ' + \
                      'Could not proceed to generate the moment maps. ' + \
                      'Please run generate_model_cube first!'
            self.log_message(err_msg)
            if not block_signal:
                self.emit_finished_with_error(err_msg)
            return
        #
        model_flux_map = None
        model_vel_map = None
        model_disp_map = None
        #
        # 20250827 new dysmalpy: gal.data becomes gal.observations['OBS'].data, and gal.instrument becomes gal.observations['OBS'].instrument
        if not hasattr(gal, 'data'): 
            if hasattr(gal, 'observations'):
                gal.data = gal.observations['OBS'].data
                gal.model_cube = gal.observations['OBS'].model_data # 20251117 the flux rescaled model cube is obs.model_data
        if not hasattr(gal, 'instrument'): 
            if hasattr(gal, 'observations'):
                gal.instrument = gal.observations['OBS'].instrument
        # 
        if gal.model_cube is None:
            err_msg = 'Error! DysmalPyGal.model_cube is invalid! ' + \
                      'Could not proceed to generate the moment maps. ' + \
                      'Please run generate_model_cube first!'
            self.log_message(err_msg)
            if not block_signal:
                self.emit_finished_with_error(err_msg)
            return
        # 
        data_mask = None
        if hasattr(gal, 'data'):
            if hasattr(gal.data, 'ndim'):
                if gal.data.ndim in [2, 3]:
                    data_mask = gal.data.mask
        #if data_mask is not None:
        #    print('***DEBUG*** np.count_nonzero(data_mask)', np.count_nonzero(data_mask), 'data_mask.shape', data_mask.shape)
        #
        model_flux_map, model_vel_map, model_disp_map = \
            self.compute_moment_maps_from_cube(params, gal.model_cube.data, data_mask)
        #
        self.model_flux_map = model_flux_map
        self.model_vel_map = model_vel_map
        self.model_disp_map = model_disp_map
        #fits.PrimaryHDU(model_flux_map).writeto('tmp.model_flux_map.fits', overwrite=True)
        # 
        #if 'vel_shift' in params: # 20230913
        #    self.model_vel_map -= params['vel_shift']
        # 
        self.send_data_to_queue(['model_flux_map', 'model_vel_map', 'model_disp_map'])
        # 
        # print('generate_moment_maps', "hasattr(gal, 'data')", hasattr(gal, 'data')) # 20250827
        # print('generate_moment_maps', "hasattr(gal, 'instrument')", hasattr(gal, 'instrument')) # 20250827
        # print('gal.data.ndim', gal.data.ndim)
        # 
        if hasattr(gal, 'data'):
            if gal.data is not None:
                self.data_flux_map = None
                self.data_vel_map = None
                self.data_disp_map = None
                self.data_flux_err_map = None
                self.data_vel_err_map = None
                self.data_disp_err_map = None
                self.data_mask_map = None
                self.residual_flux_map = None
                self.residual_vel_map = None
                self.residual_disp_map = None
                if gal.data.ndim == 3:
                    if hasattr(gal, 'data2d') and gal.data2d is not None:
                        #print('using gal.data2d')
                        self.data_flux_map = gal.data2d.data['flux']
                        self.data_vel_map = gal.data2d.data['velocity']
                        self.data_disp_map = gal.data2d.data['dispersion']
                        if hasattr(gal.data2d, 'error'):
                            self.data_flux_err_map = gal.data2d.error['flux']
                            self.data_vel_err_map = gal.data2d.error['velocity']
                            self.data_disp_err_map = gal.data2d.error['dispersion']
                    else:
                        # Get a mask2d first
                        data_mask2d = None
                        if gal.data.mask is not None:
                            if len(gal.data.mask.shape) == 3:
                                data_mask2d = np.any(gal.data.mask>0, axis=0).astype(int)
                                self.data_mask_map = data_mask2d.astype(bool)
                            elif len(gal.data.mask.shape) == 2:
                                data_mask2d = gal.data.mask
                                self.data_mask_map = data_mask2d.astype(bool)
                        # 
                        # The input data is 3d, we need to extract 2d data.
                        # We can use the
                        # `dysmalpy.plotting.extract_1D_2D_data_gausfit_from_cube`
                        # or `dysmalpy.plotting.extract_1D_2D_data_moments_from_cube`
                        # function which extracted 2d data into `gal.data2d`.
                        #
                        # gal = extract_1D_2D_data_gausfit_from_cube(gal)
                        # gal = extract_1D_2D_data_moments_from_cube(gal)
                        #
                        # These functions use following code to extract the 2d data:
                        # gal.data2d = plotting.extract_2D_gausfit_from_cube(
                        #     gal.data.data, gal, errcube=gal.data.error, inst_corr=inst_corr)
                        # gal.data2d = plotting.extract_2D_moments_from_cube(
                        #     gal.data.data, gal, inst_corr=inst_corr)
                        #
                        # however, the above method can be time consuming.
                        # Here we use our own method.
                        #scube = copy.copy(gal.data.data)
                        #scube._data[scube._data==-99] = np.nan
                        flux, vel, disp = self.compute_moment_maps_from_cube(
                            params,
                            gal.data.data, 
                            gal.data.mask, # same as data_mask
                        )
                        if flux is None or vel is None or disp is None:
                            err_msg = 'Error! Could not run compute_moment_maps_from_cube ' + \
                                      'to generate the moment maps for the data cube in 3D.'
                            self.log_message(err_msg)
                            if not block_signal:
                                self.emit_finished_with_error(err_msg)
                        # print('QDysmalPyFittingStarship generate_moment_maps '+
                        #       'flux.shape '+str(flux.shape)+
                        #       'vel.shape '+str(vel.shape)+
                        #       'disp.shape '+str(disp.shape)+
                        #       'gal.data.mask is None '+str(gal.data.mask))
                        # self.logger.debug('flux.shape '+str(flux.shape))
                        # self.logger.debug('vel.shape '+str(vel.shape))
                        # self.logger.debug('disp.shape '+str(disp.shape))
                        # 20250827 debugging
                        #print('gal.data2d = data_classes.Data2D(...)')
                        #fits.PrimaryHDU(scube).writeto('tmp.gal.data.data.fits', overwrite=True)
                        #fits.PrimaryHDU(flux).writeto('tmp.data2d.flux.fits', overwrite=True)
                        #fits.PrimaryHDU(vel).writeto('tmp.data2d.vel.fits', overwrite=True)
                        #fits.PrimaryHDU(disp).writeto('tmp.data2d.disp.fits', overwrite=True)
                        gal.data2d = data_classes.Data2D(
                            pixscale = gal.instrument.pixscale.value,
                            flux = flux, velocity = vel, vel_disp = disp, mask = data_mask2d,
                            flux_err = None, vel_err = None, vel_disp_err = None, 
                            smoothing_type = gal.data.smoothing_type,
                            smoothing_npix = gal.data.smoothing_npix,
                            inst_corr = False, moment = False,
                            xcenter = gal.data.xcenter,
                            ycenter = gal.data.ycenter,
                        )
                        self.data_flux_map = gal.data2d.data['flux']
                        self.data_vel_map = gal.data2d.data['velocity']
                        self.data_disp_map = gal.data2d.data['dispersion']
                    #
                    # if hasattr(gal, 'data1d') and gal.data1d is not None:
                    #     # TODO
                    #     raise NotImplementedError()
                    #     self.data_flux_map = gal.data2d.data['flux']
                    #     self.data_vel_map = gal.data2d.data['velocity']
                    #     self.data_disp_map = gal.data2d.data['dispersion']
                    #     if hasattr(gal.data2d, 'error'):
                    #         self.data_flux_err_map = gal.data2d.error['flux']
                    #         self.data_vel_err_map = gal.data2d.error['velocity']
                    #         self.data_disp_err_map = gal.data2d.error['dispersion']
                    # else:
                    #     # gal.data1d = extract_1D_from_cube(
                    #     #     gal.data.data, gal,
                    #     #     errcube = gal.data.error,
                    #     #     slit_width = slit_width, slit_pa = slit_pa,
                    #     #     aper_dist = aper_dist,
                    #     #     moment = False, inst_corr = inst_corr, fill_mask = fill_mask)
                    #     # gal.data1d = plotting.extract_1D_from_cube(
                    #     #     gal.data.data, gal, slit_width = slit_width,
                    #     #     slit_pa = slit_pa, aper_dist = aper_dist, moment = True,
                    #     #     inst_corr = inst_corr, fill_mask = fill_mask)
                    #     pass
                elif gal.data.ndim == 2:
                    print('gal.data.data.keys()', gal.data.data.keys())
                    self.data_flux_map = gal.data.data['flux']
                    self.data_vel_map = gal.data.data['velocity']
                    self.data_disp_map = gal.data.data['dispersion']
                    self.data_mask_map = gal.data.mask
                    if hasattr(gal.data, 'error'):
                        self.data_flux_err_map = gal.data.error['flux']
                        self.data_vel_err_map = gal.data.error['velocity']
                        self.data_disp_err_map = gal.data.error['dispersion']
                    #if 'vel_shift' in params: # 20230913
                    #    self.data_vel_map -= params['vel_shift']
                # 
                if self.data_mask_map is None: # 20250827
                    if self.data_flux_map is not None: # 20250827
                        self.data_mask_map = np.full(self.data_flux_map.shape, fill_value=True) # 20250827
                    elif self.data_vel_map is not None: # 20250827
                        self.data_mask_map = np.full(self.data_vel_map.shape, fill_value=True) # 20250827
                # 
                if self.data_flux_map is not None:
                    nan_mask = np.invert(self.data_mask_map) # (self.data_flux_map < -9.99999e5)
                    if np.count_nonzero(nan_mask)>0:
                        self.data_flux_map[nan_mask] = np.nan
                if self.data_vel_map is not None:
                    nan_mask = np.invert(self.data_mask_map) # (self.data_vel_map < -9.99999e5)
                    if np.count_nonzero(nan_mask)>0:
                        self.data_vel_map[nan_mask] = np.nan
                if self.data_disp_map is not None:
                    nan_mask = np.invert(self.data_mask_map) # (self.data_disp_map < -9.99999e5)
                    if np.count_nonzero(nan_mask)>0:
                        self.data_disp_map[nan_mask] = np.nan
                if self.data_flux_err_map is not None:
                    nan_mask = np.invert(self.data_mask_map) # (self.data_flux_err_map < -9.99999e5)
                    if np.count_nonzero(nan_mask)>0:
                        self.data_flux_err_map[nan_mask] = np.nan
                if self.data_vel_err_map is not None:
                    nan_mask = np.invert(self.data_mask_map) # (self.data_vel_err_map < -9.99999e5)
                    if np.count_nonzero(nan_mask)>0:
                        self.data_vel_err_map[nan_mask] = np.nan
                if self.data_disp_err_map is not None:
                    nan_mask = np.invert(self.data_mask_map) # (self.data_disp_err_map < -9.99999e5)
                    if np.count_nonzero(nan_mask)>0:
                        self.data_disp_err_map[nan_mask] = np.nan
                if self.data_flux_map is not None:
                    if self.data_flux_map.shape == self.model_flux_map.shape:
                        self.residual_flux_map = self.data_flux_map - self.model_flux_map
                if self.data_vel_map is not None:
                    if self.data_vel_map.shape == self.model_vel_map.shape:
                        self.residual_vel_map = self.data_vel_map - self.model_vel_map
                if self.data_disp_map is not None:
                    if self.data_disp_map.shape == self.model_disp_map.shape:
                        self.residual_disp_map = self.data_disp_map - self.model_disp_map
                #
                self.send_data_to_queue(
                        ['data_flux_map', 'data_vel_map', 'data_disp_map', 'data_mask_map', 
                         'data_flux_err_map', 'data_vel_err_map', 'data_disp_err_map', 
                         'residual_flux_map', 'residual_vel_map', 'residual_disp_map']
                    )
        #
        self.logger.debug('generateMomentMaps: done')
        self.log_message('Successfully generated moment maps.')
        if not block_signal:
            self.emit_finished()
    
    #
    def generate_rotation_curves(self, params, gal = None, block_signal = False):
        if not block_signal:
            self.emit_started()
        #
        self.log_message('Generating rotation curves...')
        #
        if params is None:
            err_msg = 'Error! DysmalPyParams is invalid! ' + \
                      'Could not proceed to generate the rotation curve. ' + \
                      'Please set DysmalPyFittingWorker.DysmalPyParams first!'
            self.log_message(err_msg)
            if not block_signal:
                self.emit_finished_with_error(err_msg)
            return
        if gal is None:
            gal = self.gal
        if gal is None:
            err_msg = 'Error! DysmalPyGal is invalid! ' + \
                      'Could not proceed to generate the rotation curve. ' + \
                      'Please run DysmalPyFittingWorker.generateModelCube first!'
            self.log_message(err_msg)
            if not block_signal:
                self.emit_finished_with_error(err_msg)
            return
        if gal.model_cube is None:
            err_msg = 'Error! DysmalPyGal.model_cube is invalid! ' + \
                      'Could not proceed to generate the rotation curve. ' + \
                      'Please run DysmalPyFittingWorker.generateModelCube first!'
            self.log_message(err_msg)
            if not block_signal:
                self.emit_finished_with_error(err_msg)
            return
        #
        self.data_flux_curve = None
        self.data_vel_curve = None
        self.data_disp_curve = None
        self.model_flux_curve = None
        self.model_vel_curve = None
        self.model_disp_curve = None
        inst_corr = True
        #
        kwargs_galmodel = setup_lensing_dict(params)
        kwargs_galmodel['lensing_transformer'] = self.lensing_transformer
        for key in ['moment_calc', 'gauss_extract_with_c', 'zcalc_truncate', 'zcalc_with_c']:
            if key in params:
                kwargs_galmodel[key] = params[key] #20211111
                print(f'kwargs_galmodel[{key!r}] = params[{key!r}] = {params[key]!r}') #20211111 DEBUG
        #
        # Get 1d flux, vel and disp from model.
        # If input data is 1d, i.e., having 'fdata' in params, then we can directly use the
        #   dymalpy.galaxy.create_model_data(ndim_final=1, from_data=True)
        # funcion,
        # otherwise we create a copy of dymalpy.galaxy object, then use the
        #   dymalpy.galaxy.create_model_data(ndim_final=1, from_data=False, from_instrument=True)
        # function.
        # 
        # 20250827 
        # In new dysmalpy, there is no 'create_model_data', we need to use 'create_single_obs_model_data'
        # 
        # get data 1d and instrument 1d object in new dysmalpy 20250827
        self.log_message('Computing rotation velocity profile along the data slit at PA '+str(params['slit_pa']))
        if 'fdata' in params: # the data is already 1D data
            gal_data = obs.data
            gal_model_data = obs.model_data
            data_1d = obs.data
            instrument_1d = obs.instrument
        else:
            # the data is 2D or 3D, we need to measure 1D profile for the data
            # extract 1D profiles from the 2D data
            naper = 35
            fov_npix = params['fov_npix']
            pixscale = params['pixscale']
            if 'slit_width' in params:
                slit_width = params['slit_width']
            else:
                slit_width = 0.2 #<TODO># arcsec
            # 
            if 'slit_pa' in params:
                slit_pa = params['slit_pa']
            else:
                slit_pa = 45. #<TODO># degree
            # 
            if 'profile1d_type' in params:
                profile1d_type = params['profile1d_type']
            else:
                profile1d_type = 'rect_ap_cube'
            # 
            if 'aperture_radius' in params:
                aperture_radius = params['aperture_radius']
            else:
                aperture_radius = float(slit_width / 2.0) # arcsec
            # 
            if 'pix_perp' in params:
                pix_perp = params['pix_perp']
            else:
                pix_perp = float(slit_width / pixscale)
            # 
            if 'pix_parallel' in params:
                pix_parallel = params['pix_parallel']
            else:
                pix_parallel = 1
            # 
            if 'pix_length' in params:
                pix_length = params['pix_length']
            else:
                pix_length = int(np.ceil(slit_width / pixscale))
            # 
            if 'xcenter' in params:
                xcenter = params['xcenter']
            else:
                xcenter = None
            # 
            if 'ycenter' in params:
                ycenter = params['ycenter']
            else:
                ycenter = None
            # 
            if slit_pa is None or slit_width is None:
                err_msg = 'Error! Invalid slit_width or slit_pa. ' + \
                          'Could not proceed to generate the rotation curves.'
                self.log_message(err_msg)
                if not block_signal:
                    self.emit_finished_with_error(err_msg)
                return
            # 
            fov_arcsec = np.abs(fov_npix*pixscale)
            aper_centers = np.linspace(-fov_arcsec/2.0, fov_arcsec/2.0, num=naper, endpoint=True) # in arcsec
            # 
            # prepare output arrays
            data_aper_fluxes = np.full(naper, fill_value=np.nan)
            data_aper_vels = np.full(naper, fill_value=np.nan)
            data_aper_disps = np.full(naper, fill_value=np.nan)
            data_aper_flux_errors = np.full(naper, fill_value=np.nan)
            data_aper_vel_errors = np.full(naper, fill_value=np.nan)
            data_aper_disp_errors = np.full(naper, fill_value=np.nan)
            model_aper_fluxes = np.full(naper, fill_value=np.nan)
            model_aper_vels = np.full(naper, fill_value=np.nan)
            model_aper_disps = np.full(naper, fill_value=np.nan)
            # 
            # extract 1D from 2D
            if xcenter is None:
                xcenter = (fov_npix-1.)/2.
            if ycenter is None:
                ycenter = (fov_npix-1.)/2.
            ygrid, xgrid = np.mgrid[0:fov_npix, 0:fov_npix]
            xvec = np.arange(fov_npix)
            yvec = np.arange(fov_npix)
            rect_height = pix_parallel
            rect_width = pix_perp
            rect_b_a_angle = np.arctan2(rect_height/2.0, rect_width/2.0) # degree
            rect_half_diagonal = np.sqrt((rect_height/2.0)**2 + (rect_width/2.0)**2) # arcsec
            rect_angle = slit_pa+90.+180. # rect_width, rect_height, rect_angle (up from +x), 
            if rect_angle > 180.:
                rect_angle -= 360.
            if rect_angle < -180.:
                rect_angle += 360.
            aper_centers_blue_to_red = aper_centers
            aper_rarr = aper_centers_blue_to_red
            # -180. accounts for the fact that we want apertures to scan from blue to red, 
            # but DysmalPy defines PA as the blue side direction from North.  
            for iaper in range(naper):
                xpos = aper_centers_blue_to_red[iaper] * np.cos(np.deg2rad(rect_angle)) / pixscale + xcenter
                ypos = aper_centers_blue_to_red[iaper] * np.sin(np.deg2rad(rect_angle)) / pixscale + ycenter
                if profile1d_type == 'circ_ap_cube' or profile1d_type == 'circ_ap_pv':
                    # circular apertures
                    grid_mask = (np.sqrt((xgrid-xpos)**2 + (ygrid-ypos)**2) <= (aperture_radius / pixscale)) # True for inside the circlar aperture
                elif profile1d_type == 'single_pix_pv':
                    # single-pixel apertures
                    grid_mask = np.full(xgrid.shape, fill_value=False)
                    xposi, yposi = int(np.round(xpos)), int(np.round(ypos))
                    if xposi >= 0 and xposi <= fov_npix-1 and \
                       yposi >= 0 and yposi <= fov_npix-1 :
                        grid_mask[yposi, xposi] = True
                else:
                    # rectangle apertures
                    angle11 = np.deg2rad(rect_angle+90.+rect_b_a_angle)
                    angle12 = np.deg2rad(rect_angle+90.-rect_b_a_angle)
                    angle22 = np.deg2rad(rect_angle-90.+rect_b_a_angle)
                    angle21 = np.deg2rad(rect_angle-90.-rect_b_a_angle)
                    xpos11 = xpos + rect_half_diagonal * np.cos(angle11) / pixscale
                    ypos11 = ypos + rect_half_diagonal * np.sin(angle11) / pixscale
                    xpos12 = xpos + rect_half_diagonal * np.cos(angle12) / pixscale
                    ypos12 = ypos + rect_half_diagonal * np.sin(angle12) / pixscale
                    xpos22 = xpos + rect_half_diagonal * np.cos(angle22) / pixscale
                    ypos22 = ypos + rect_half_diagonal * np.sin(angle22) / pixscale
                    xpos21 = xpos + rect_half_diagonal * np.cos(angle21) / pixscale
                    ypos21 = ypos + rect_half_diagonal * np.sin(angle21) / pixscale
                    #if iaper == naper//2:
                    #    print('dzliu debugging: xcenter, ycenter:', 
                    #          (xcenter, ycenter))
                    #    print('dzliu debugging: xpos11, ypos11, xpos12, ypos12, xpos22, ypos22, xpos21, ypos21:', 
                    #          (xpos11, ypos11), (xpos12, ypos12), (xpos22, ypos22), (xpos21, ypos21))
                    # test point in rectangle, test point at the right side of each line from (1,1)-(1,2)-(2,2)-(2,1)
                    # xpos12, ypos12, xpos11, ypos11, xgrid, ygrid = 5, 5, 4, 4, 5, 4
                    grid_mask = np.full(xgrid.shape, fill_value=False) # True for outside the aper step rect
                    for checkpoints in [ [ (xpos11,ypos11), (xpos12,ypos12) ], 
                                         [ (xpos12,ypos12), (xpos22,ypos22) ], 
                                         [ (xpos22,ypos22), (xpos21,ypos21) ], 
                                         [ (xpos21,ypos21), (xpos11,ypos11) ] ]:
                        checkpoint_a, checkpoint_b = checkpoints
                        xa, ya = checkpoint_a
                        xb, yb = checkpoint_b
                        check_mask = ((xb-xa)*(ygrid-ya) - (yb-ya)*(xgrid-xa) < 0.0) # (xgrid,ygrid) is at the left side of line (xa,ya)-(xb,yb)
                        grid_mask = np.logical_or(grid_mask, check_mask)
                    grid_mask = np.invert(grid_mask) # now True for inside the rect
                    # # 
                    # #if iaper == naper//2:
                    # #    print('dzliu debugging: np.count_nonzero(check_mask):', 
                    # #          np.count_nonzero(check_mask))
                    # if np.count_nonzero(grid_mask) == 0:
                    #     xposi, yposi = int(np.round(xpos)), int(np.round(ypos))
                    #     if xposi >= 0 and xposi <= fov_npix-1 and \
                    #        yposi >= 0 and yposi <= fov_npix-1 :
                    #         grid_mask[yposi, xposi] = True
                # 
                if np.count_nonzero(grid_mask) > 0:
                    has_data_1d_measure = False
                    if self.data_flux_map is not None:
                        data_aper_fluxes[iaper] = np.nanmean(self.data_flux_map[grid_mask])
                        has_data_1d_measure = True
                    if self.data_vel_map is not None:
                        data_aper_vels[iaper] = np.nanmean(self.data_vel_map[grid_mask])
                        has_data_1d_measure = True
                    if self.data_disp_map is not None:
                        data_aper_disps[iaper] = np.nanmean(self.data_disp_map[grid_mask])
                        has_data_1d_measure = True
                    if self.data_flux_err_map is not None:
                        data_aper_flux_errors[iaper] = np.nanmean(self.data_flux_err_map[grid_mask])
                        has_data_1d_measure = True
                    if self.data_vel_err_map is not None:
                        data_aper_vel_errors[iaper] = np.nanmean(self.data_vel_err_map[grid_mask])
                        has_data_1d_measure = True
                    if self.data_disp_err_map is not None:
                        data_aper_disp_errors[iaper] = np.nanmean(self.data_disp_err_map[grid_mask])
                        has_data_1d_measure = True
                    # 
                    has_model_1d_measure = False
                    if self.model_flux_map is not None:
                        model_aper_fluxes[iaper] = np.nanmean(self.model_flux_map[grid_mask])
                        has_model_1d_measure = True
                    if self.model_vel_map is not None:
                        model_aper_vels[iaper] = np.nanmean(self.model_vel_map[grid_mask])
                        has_model_1d_measure = True
                    if self.model_disp_map is not None:
                        model_aper_disps[iaper] = np.nanmean(self.model_disp_map[grid_mask])
                        has_model_1d_measure = True
                    # 
                    if not has_data_1d_measure:
                        if self.data_cube is not None:
                            spec_axis = self.data_cube.spectral_axis.to(u.km/u.s).value
                            spec_flux = np.array([np.nanmean(chanimg[grid_mask]) for chanimg in self.data_cube._data])
                            spec_mom0 = np.abs(np.nanmean(spec_flux))
                            spec_mom1 = np.nansum(spec_flux*spec_axis)/np.nansum(spec_flux)
                            spec_mom2 = np.sqrt(np.abs(np.nansum(spec_flux*spec_axis**2)/np.nansum(spec_flux)))
                            best_fit = gaus_fit_sp_opt_leastsq(spec_axis, spec_flux, 
                                spec_mom0 / np.sqrt(2 * np.pi) / np.abs(spec_mom2), spec_mom1, spec_mom2)
                            data_aper_fluxes[iaper] = best_fit[0] * np.sqrt(2 * np.pi) * best_fit[2]
                            data_aper_vels[iaper] = best_fit[1]
                            data_aper_disps[iaper] = best_fit[2]
                            data_aper_flux_errors[iaper] = np.abs(data_aper_fluxes[iaper]) * 0.1 # TODO: error for the spec 1d
                            data_aper_vel_errors[iaper] = 5.0 # TODO: error for the spec 1d
                            data_aper_disp_errors[iaper] = 5.0 # TODO: error for the spec 1d
                    # 
                    if not has_model_1d_measure:
                        if self.model_cube is not None:
                            spec_axis = self.model_cube.spectral_axis.to(u.km/u.s).value
                            spec_flux = np.array([np.nanmean(chanimg[grid_mask]) for chanimg in self.model_cube._data])
                            spec_mom0 = np.abs(np.nanmean(spec_flux))
                            spec_mom1 = np.nansum(spec_flux*spec_axis)/np.nansum(spec_flux)
                            spec_mom2 = np.sqrt(np.abs(np.nansum(spec_flux*spec_axis**2)/np.nansum(spec_flux)))
                            best_fit = gaus_fit_sp_opt_leastsq(spec_axis, spec_flux, spec_mom0, spec_mom1, spec_mom2)
                            model_aper_fluxes[iaper] = best_fit[0] * np.sqrt(2 * np.pi) * best_fit[2]
                            model_aper_vels[iaper] = best_fit[1]
                            model_aper_disps[iaper] = best_fit[2]
                
                # # extract 1D from 3D
                # DummyObs = namedtuple('DummyObs', ['instrument', 'obs_options'])
                # DummyInst = namedtuple('DummyInst', ['pixscale', 'fov', 'moment'])
                # DummyOpts = namedtuple('DummyOpts', ['xcenter', 'ycenter'])
                # dummy_inst = DummyInst(pixscale, [fov_npix, fov_npix], params['moment_calc'])
                # dummy_opts = DummyOpts(params['xcenter'], params['ycenter'])
                # dummy_obs = DummyObs(dummy_inst, dummy_opts)

                # apertures = aperture_classes.setup_aperture_types( # 20250827 new dysmalpy
                #     obs=dummy_obs,
                #     profile1d_type=profile1d_type,
                #     aper_centers=aper_centers,
                #     aperture_radius=aperture_radius,
                #     slit_pa=slit_pa,
                #     slit_width=slit_width,
                #     pix_perp=pix_perp, 
                #     pix_parallel=pix_parallel,
                #     partial_weight=partial_weight,
                # )

                # aper_centers, flux1d, vel1d, disp1d = apertures.extract_1d_kinematics(
                #     spec_arr=vel_arr,
                #     cube=data_scaled, mask=mask, err=ecube,
                #     center_pixel = center_pixel, pixscale=pixscale,
                # )
                # aper_fluxes = flux1d.data
                # aper_flux_errors = flux1d.error
                # aper_vels = vel1d.data
                # aper_vel_errors = vel1d.error
                # aper_disps = disp1d.data
                # aper_disp_errors = disp1d.error
        # 
        if 'data_inst_corr' in params:
            inst_corr = params['data_inst_corr']
        else:
            inst_corr = False
        gal_data = data_classes.Data1D(r=aper_centers, velocity=data_aper_vels, vel_err=data_aper_vel_errors, 
            vel_disp=data_aper_disps, vel_disp_err=data_aper_disp_errors, 
            flux=data_aper_fluxes, flux_err=data_aper_flux_errors, 
            inst_corr=inst_corr)
        gal_model_data = data_classes.Data1D(r=aper_centers, velocity=model_aper_vels, 
            vel_disp=model_aper_disps, 
            flux=model_aper_fluxes, 
            inst_corr=inst_corr)
        # 
        self.data_flux_curve  = {'x':gal_data.rarr,
                                 'y':gal_data.data['flux'],
                                 'yerr':gal_data.error['flux'],
                                 'marker':'o', 'markersize':5, 'markeredgecolor':'none',
                                 'linestyle':'none', 'alpha':0.7, 'capsize':2}
        self.data_vel_curve   = {'x':gal_data.rarr,
                                 'y':gal_data.data['velocity'],
                                 'yerr':gal_data.error['velocity'],
                                 'marker':'o', 'markersize':5, 'markeredgecolor':'none',
                                 'linestyle':'none', 'alpha':0.7, 'capsize':2}
        self.data_disp_curve  = {'x':gal_data.rarr,
                                 'y':gal_data.data['dispersion'],
                                'yerr':gal_data.error['dispersion'],
                                'marker':'o', 'markersize':5, 'markeredgecolor':'none',
                                 'linestyle':'none', 'alpha':0.7, 'capsize':2}
        self.model_flux_curve = {'x':gal_model_data.rarr,
                                 'y':gal_model_data.data['flux'],
                                 'marker':'s', 'markersize':5, 'markeredgecolor':'none',
                                 'linestyle':'none', 'alpha':0.7}
        self.model_vel_curve  = {'x':gal_model_data.rarr,
                                 'y':gal_model_data.data['velocity'],
                                 'marker':'s', 'markersize':5, 'markeredgecolor':'none',
                                 'linestyle':'none', 'alpha':0.7}
        self.model_disp_curve = {'x':gal_model_data.rarr,
                                 'y':gal_model_data.data['dispersion'],
                                 'marker':'s', 'markersize':5, 'markeredgecolor':'none',
                                 'linestyle':'none', 'alpha':0.7}
        
        # if 'inst_corr' in gal_data.data.keys():
        #     inst_corr = gal_data.data['inst_corr']
        if inst_corr:
            try:
                lsf_dispersion = gal_instrument.lsf.dispersion
            except:
                self.log_message('LSF dispersion not defined. Will not do inst_corr.')
                inst_corr = False
        if inst_corr:
            self.model_disp_curve['y'] = np.sqrt( self.model_disp_curve['y']**2 - lsf_dispersion.to(u.km/u.s).value**2 )
            self.log_message('LSF correction done.')
            # see "dysmalpy/plotting.py" def plot_data_model_comparison_1D
        #
        self.send_data_to_queue(
                ['data_flux_curve', 'data_vel_curve', 'data_disp_curve', 
                 'model_flux_curve', 'model_vel_curve', 'model_disp_curve']
            )
        #
        self.logger.debug('generateRotationCurves: done')
        self.log_message('Successfully generated rotation curves.')
        if not block_signal:
            self.emit_finished()
    
    # 
    #def more



    


#

class QWidgetForParamInput(QWidget):
    """QWidget for a param input.
    
    The widget can be a QCheckBox, if the parameter has a boolean data type,
    or QComboBox, if the parameter has multiple options or is a boolean data type
    but with checkbox = False, or more commonly, the widget is a QLineEdit,
    and parameter data type can be a list or a str or a int or a float.
    """
    
    ParamUpdateSignal = pyqtSignal(str, str, type, type)
    
    def __init__(self,
            keyname, keyvalue=None, keycomment='', datatype=str, listtype=str,
            default=None, options=None, checkbox=False, 
            fullwidth=False, # whether this widget span the whole width. in default it is half width.
            isdatadir=False, defaultdir=None, isdatafile=False, namefilter=None, isoutdir=False,
            readonly=False, enabled=True, 
            associatedparams=None, 
            parent=None,
        ):
        super(QWidgetForParamInput, self).__init__()
        self.ParamName = keyname
        self.ParamValue = keyvalue # note that this will change if the user edited anything via the GUI
        self.ParamComment = keycomment
        self.ParamDefaultValue = default
        self.ParamDataType = datatype
        self.ParamListType = listtype # indicate whether the value should be a list or not. listtype == list.
        self.ParamRegExpValidator = ''
        self.LabelText = keyname
        self.IsFullWidth = fullwidth
        self.IsDataDir = isdatadir
        self.IsDataFile = isdatafile
        self.IsOutDir = isoutdir
        self.NameFilter = namefilter
        self.DefaultDirectory = defaultdir if defaultdir is not None else os.getcwd()
        self.RestrictedToDirectory = None
        self.AssociatedParams = associatedparams
        self.IndexInWidgetGrid = -1
        self.LabelWidget = None
        self.LineEditWidget = None
        self.CheckBoxWidget = None
        self.ComboBoxWidget = None
        self.ButtonToOpen = None
        self.ButtonToEdit = None
        self.ButtonToClose = None
        self.Layout = QHBoxLayout()
        self.Layout.setContentsMargins(0, 0, 0, 0)
        self.Layout.setSpacing(0)
        if self.LabelText.startswith('__'):
            if re.match(r'__(.*)__', self.LabelText):
                self.LabelText = re.sub(r'__(.*)__', r'\1', self.LabelText)
            else:
                self.LabelText = re.sub(r'__(.*)', r'\1', self.LabelText)
        if self.ParamDataType is bool and self.ParamListType is not list and checkbox == True:
            # checkbox for bool
            self.CheckBoxWidget = QCheckBox(self.LabelText)
            self.CheckBoxWidget.setMinimumWidth(160)
            self.CheckBoxWidget.setToolTip(self.ParamComment)
            self.Layout.addWidget(self.CheckBoxWidget)
            self.CheckBoxWidget.stateChanged.connect(self.onCheckBoxStateChangedCall)
        elif self.ParamDataType is bool and self.ParamListType is not list:
            # combobox for bool
            self.LabelWidget = QLabel(self.LabelText+':')
            self.LabelWidget.setMinimumWidth(80)
            self.ComboBoxWidget = QComboBox()
            self.ComboBoxWidget.setToolTip(self.ParamComment)
            self.ComboBoxWidget.addItem(self.tr('True'))
            self.ComboBoxWidget.addItem(self.tr('False'))
            self.Layout.addWidget(self.LabelWidget)
            self.Layout.addWidget(self.ComboBoxWidget)
            self.ComboBoxWidget.currentIndexChanged.connect(self.onComboBoxIndexChangedCall)
        elif options is not None:
            # combobox for options
            self.LabelWidget = QLabel(self.LabelText+':')
            self.LabelWidget.setMinimumWidth(80)
            self.ComboBoxWidget = QComboBox()
            self.ComboBoxWidget.setToolTip(self.ParamComment)
            for this_option in options:
                self.ComboBoxWidget.addItem(this_option)
            self.Layout.addWidget(self.LabelWidget)
            self.Layout.addWidget(self.ComboBoxWidget)
            self.ComboBoxWidget.currentIndexChanged.connect(self.onComboBoxIndexChangedCall)
        else:
            self.LabelWidget = QLabel(self.LabelText+':')
            self.LabelWidget.setMinimumWidth(80)
            self.LineEditWidget = QLineEdit()
            self.LineEditWidget.setToolTip(self.ParamComment)
            self.LineEditWidget.setReadOnly(readonly)
            self.Layout.addWidget(self.LabelWidget)
            self.Layout.addWidget(self.LineEditWidget)
            if self.ParamDataType is not None:
                if self.ParamDataType is bool:
                    self.ParamRegExpValidator = r'(True|False)'
                elif self.ParamDataType is int:
                    self.ParamRegExpValidator = r'([+-]|)[0-9.]+'
                elif self.ParamDataType is float:
                    self.ParamRegExpValidator = r'([+-]|)[0-9.]+([eE][0-9]+|[eE][+-][0-9]+)'
                else:
                    pass
            if self.ParamListType is list:
                if self.ParamRegExpValidator == '':
                    self.ParamRegExpValidator = r'\[.*\]'
                else:
                    self.ParamRegExpValidator = r'\['+self.ParamRegExpValidator+r', *'+self.ParamRegExpValidator+r'.*'+r'\]'
            if self.ParamRegExpValidator != '':
                self.LineEditWidget.setValidator(QRegExpValidator(QRegExp(r'^'+self.ParamRegExpValidator+r'$')))
            self.LineEditWidget.setStyleSheet("""
                QLineEdit {
                    background: white;
                    border: 1px solid #cccccc;
                    border-radius: 3px;
                }
                QLineEdit[readOnly="true"] {
                    background: #dedede;
                    border: 1px solid #cccccc;
                }
                QLineEdit[enabled="false"] {
                    background: #dedede;
                    border: 1px solid #cccccc;
                }
                """)
            self.LineEditWidget.textChanged.connect(self.onLineEditTextChangedCall)
            #
            if self.IsDataDir or self.IsDataFile or self.IsOutDir:
                self.ButtonToOpen = QPushButton(self.tr('...'))
                self.ButtonToEdit = QPushButton(self.tr('✓'))
                self.ButtonToClose = QPushButton(self.tr('✘'))
                if self.IsDataDir:
                    self.ButtonToOpen.setToolTip('Select a data directory.')
                elif self.IsDataFile:
                    self.ButtonToOpen.setToolTip('Select a data file.')
                elif self.IsOutDir:
                    self.ButtonToOpen.setToolTip('Select an output directory.')
                self.ButtonToEdit.setToolTip('Edit')
                self.ButtonToClose.setToolTip('Close')
                self.ButtonToOpen.clicked.connect(self.onButtonToOpenCall)
                self.ButtonToEdit.clicked.connect(self.onButtonToEditCall)
                self.ButtonToClose.clicked.connect(self.onButtonToCloseCall)
                self.Layout.addWidget(self.ButtonToOpen)
                self.Layout.addWidget(self.ButtonToEdit)
                self.Layout.addWidget(self.ButtonToClose)
                self.setAcceptDrops(True)
            #
        self.setLayout(self.Layout)
        #
        if self.ParamDefaultValue is not None:
            self.setText(self.ParamDefaultValue)
        #
        if not enabled:
            self.setEnabled(enabled)

    def getPositionInQGridLayout(self, icount, ncolumn):
        """Return a position to add this widget to a QGridLayout.
        
        The input is the searlized index `icount` in the QGridLayout and the grid column width `ncolumn`.
        
        The output is the incremented searlized index for the next widget in the QGridLayout,
        the row index, column index, row span and column span of this widget, which are to be
        used when calling the QGridLayout.addWidget() function.
        """
        irow = int(icount / ncolumn)
        icol = (icount % ncolumn)
        rowSpan = 1
        colSpan = 1
        # if the widget is set to be full width, then we need to make sure its icol is 0 and colSpan is ncolumn
        if self.IsFullWidth:
            if icol != 0:
                irow += 1
                icount += ncolumn-1-icol+1
            colSpan = ncolumn
        icount += colSpan
        return icount, irow, icol, rowSpan, colSpan

    def isFullWidth(self):
        return self.IsFullWidth

    def keyname(self):
        return self.ParamName

    def keyvalue(self):
        #return self.ParamValue
        return self.textToValue(self.text())

    def keycomment(self):
        return self.ParamComment
    
    def textToValue(self, text):
        keyvalue = None
        if text == '':
            keyvalue = None
        elif text == 'None':
            keyvalue = None
        elif text == 'True':
            keyvalue = True
        elif text == 'False':
            keyvalue = False
        elif text == 'inf':
            keyvalue = np.inf
        elif text == 'nan':
            keyvalue = np.nan
        else:
            if self.ParamListType is list:
                keyvalue = eval(text)
                keyvalue = np.array(keyvalue).astype(self.ParamDataType)
                # note that here may cause exception, better to use try except when calling this function
            else:
                keyvalue = np.array([text]).astype(self.ParamDataType)[0]
                # note that here may cause exception, better to use try except when calling this function
        return keyvalue

    def text(self):
        if self.LineEditWidget is not None:
            return self.LineEditWidget.text()
        elif self.ComboBoxWidget is not None:
            return self.ComboBoxWidget.currentText()
        elif self.CheckBoxWidget is not None:
            if self.CheckBoxWidget.isChecked():
                return 'True'
            else:
                return 'False'
        else:
            return ''

    def default(self):
        return self.ParamDefaultValue

    def toolTip(self):
        if self.LineEditWidget is not None:
            return self.LineEditWidget.toolTip()
        elif self.ComboBoxWidget is not None:
            return self.ComboBoxWidget.toolTip()
        elif self.CheckBoxWidget is not None:
            return self.CheckBoxWidget.toolTip()
        else:
            return ''
    
    def dataType(self):
        return self.ParamDataType
    
    def listType(self):
        return self.ParamListType
    
    def setText(self, text, blocksignal=False):
        if text is None:
            text = ''
        if self.LineEditWidget is not None:
            if blocksignal:
                self.LineEditWidget.blockSignals(True)
            if text != '' and (not self.isEnabled()):
                self.setEnabled(True)
            self.LineEditWidget.setText(str(text))
            if blocksignal:
                self.LineEditWidget.blockSignals(False)
        elif self.ComboBoxWidget is not None:
            found_index = self.ComboBoxWidget.findText(str(text))
            if found_index >= 0:
                if blocksignal:
                    self.ComboBoxWidget.blockSignals(True)
                self.ComboBoxWidget.setCurrentIndex(found_index)
                if blocksignal:
                    self.ComboBoxWidget.blockSignals(False)
        elif self.CheckBoxWidget is not None:
            if blocksignal:
                self.CheckBoxWidget.blockSignals(True)
            if self.CheckBoxWidget.isChecked() and text == 'False':
                self.CheckBoxWidget.setChecked(False)
            elif not self.CheckBoxWidget.isChecked() and text == 'True':
                self.CheckBoxWidget.setChecked(True)
            if blocksignal:
                self.CheckBoxWidget.blockSignals(False)
    
    def setChecked(self, checked, blocksignal=False):
        if self.CheckBoxWidget is not None:
            if blocksignal:
                self.CheckBoxWidget.blockSignals(True)
            self.CheckBoxWidget.setChecked(checked)
            if blocksignal:
                self.CheckBoxWidget.blockSignals(False)

    def setEnabled(self, enabled):
        #logger.debug('QWidgetForParamInput::setEnabled({})'.format(enabled))
        if self.isEnabled() == enabled:
            return
        if self.LineEditWidget is not None:
            #self.LineEditWidget.setEnabled(enabled)
            self.LineEditWidget.setReadOnly(operator.not_(enabled))
            #logger.debug('QWidgetForParamInput::setEnabled() self.LineEditWidget.setReadOnly({})'.format(operator.not_(enabled)))
            self.LineEditWidget.style().unpolish(self.LineEditWidget)
            self.LineEditWidget.style().polish(self.LineEditWidget)

    def isEnabled(self):
        if self.LineEditWidget is not None:
            #return self.LineEditWidget.isEnabled()
            return operator.not_(self.LineEditWidget.isReadOnly())
        return True

    def setEditable(self, editable):
        if self.LineEditWidget is not None:
            self.LineEditWidget.setReadOnly(operator.not_(editable))
            self.LineEditWidget.style().unpolish(self.LineEditWidget)
            self.LineEditWidget.style().polish(self.LineEditWidget)

    def setReadOnly(self, readonly):
        if self.LineEditWidget is not None:
            self.LineEditWidget.setReadOnly(readonly)
            self.LineEditWidget.style().unpolish(self.LineEditWidget)
            self.LineEditWidget.style().polish(self.LineEditWidget)
        elif self.CheckBoxWidget is not None:
            self.CheckBoxWidget.setReadOnly(readonly)

    @pyqtSlot(str)
    def onLineEditTextChangedCall(self, updatedtext):
        #logger.debug('QWidgetForParamInput::onLineEditTextChangedCall')
        if self.LineEditWidget is not None:
            self.ParamUpdateSignal.emit(self.ParamName, updatedtext, self.ParamDataType, self.ParamListType)
    
    @pyqtSlot(int)
    def onComboBoxIndexChangedCall(self, state):
        if self.ComboBoxWidget is not None:
            self.ParamUpdateSignal.emit(self.ParamName, self.ComboBoxWidget.currentText(), self.ParamDataType, self.ParamListType)

    @pyqtSlot(int)
    def onCheckBoxStateChangedCall(self, state):
        if self.CheckBoxWidget is not None:
            if self.CheckBoxWidget.isChecked():
                self.ParamUpdateSignal.emit(self.ParamName, 'True', self.ParamDataType, self.ParamListType)
            else:
                self.ParamUpdateSignal.emit(self.ParamName, 'False', self.ParamDataType, self.ParamListType)
    
    def setRestrictedToDirectory(self, dirpath):
        if dirpath is None or dirpath == '':
            self.RestrictedToDirectory = None
        else:
            self.RestrictedToDirectory = dirpath
    
    @pyqtSlot()
    def onButtonToOpenCall(self):
        logger.debug('QWidgetForParamInput::onButtonToOpenCall()')
        #self.logMessage(self.tr('Selecting a filepath...'))
        if self.ButtonToOpen is None:
            return
        if self.LineEditWidget is None:
            return
        if self.IsDataDir or self.IsOutDir:
            dialog = QFileDialog(self, self.ButtonToOpen.toolTip(), self.DefaultDirectory, self.NameFilter)
            dialog.setFileMode(QFileDialog.DirectoryOnly) # select only directories
            if self.IsDataDir:
                dialog.setOption(QFileDialog.ReadOnly, True)
        elif self.IsDataFile:
            if self.RestrictedToDirectory is not None:
                # here we restrict to select only data files in self.DefaultDirectory
                dialog = QFileDialogRestrictedToDirectory(self, self.ButtonToOpen.toolTip(), self.RestrictedToDirectory, self.NameFilter)
            else:
                dialog = QFileDialog(self, self.ButtonToOpen.toolTip(), self.DefaultDirectory, self.NameFilter)
            dialog.setFileMode(QFileDialog.ExistingFiles)
            dialog.setOption(QFileDialog.ReadOnly, True)
        else:
            dialog = QFileDialog(self, self.ButtonToOpen.toolTip(), self.DefaultDirectory, self.NameFilter)
        #dialog.setSidebarUrls([QUrl.fromLocalFile(self.DefaultDirectory)])
        dialog.RestoreDirectory = True
        filepath = None
        if dialog.exec_() == QDialog.Accepted:
            filepath = dialog.selectedFiles()[0]
        del dialog
        if filepath is None or filepath == '':
            #self.logMessage(self.tr('No directory selected.'))
            #if not self.isEnabled():
            #    filepath = ''
            #    self.setText('', blocksignal=True)
            #    self.setEnabled(False)
            pass
        else:
            if self.IsDataDir or self.IsOutDir:
                if not filepath.endswith(os.sep):
                    filepath += os.sep
            elif self.IsDataFile:
                filepath = os.path.basename(filepath)
                #<TODO># how to make sure data files are under 'datadir'?
            #self.logMessage(self.tr('Selected data file: '+str(filepath)))
            #if not self.isEnabled():
            #    self.setEnabled(True)
            self.setText(filepath, blocksignal=True)
        self.ParamUpdateSignal.emit(self.ParamName, filepath, self.ParamDataType, self.ParamListType)
    
    @pyqtSlot()
    def onButtonToEditCall(self):
        logger.debug('QWidgetForParamInput::onButtonToEditCall()')
        self.setEnabled(True)
    
    @pyqtSlot()
    def onButtonToCloseCall(self):
        logger.debug('QWidgetForParamInput::onButtonToCloseCall()')
        self.setText('', blocksignal=True)
        self.setEnabled(False)
        self.ParamUpdateSignal.emit(self.ParamName, '', self.ParamDataType, self.ParamListType)

    @pyqtSlot(str, str, type, type)
    def onDataDirParamUpdateCall(self, keyname, keyvalue, datatype, listtype):
        logger.debug('QWidgetForParamInput::onDataDirParamUpdateCall()')
        if keyname == 'datadir':
            if self.IsDataFile:
                logger.debug('QWidgetForParamInput::onDataDirParamUpdateCall() self.setRestrictedToDirectory("{}")'.format(keyvalue))
                self.setRestrictedToDirectory(keyvalue)
    
    def dragEnterEvent(self, event):
        if self.isEnabled() and (self.IsDataDir or self.IsDataFile or self.IsOutDir):
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        if self.isEnabled() and (self.IsDataDir or self.IsDataFile or self.IsOutDir):
            #input_text = event.mimeData().text()
            #self.LineEditWidget.setText(input_text)
            for url in event.mimeData().urls():
                filepath = url.toLocalFile()
                if self.IsDataFile:
                    filepath = os.path.basename(filepath) # only take file basename
                elif not filepath.endswith(os.sep):
                    filepath += os.sep # append trailing os.sep to datadir or outdir
                self.LineEditWidget.setText(filepath)
                break
            event.accept()





class QFileDialogRestrictedToDirectory(QFileDialog):
    """A QFileDialog that selects files restricted in a given directory."""
    def __init__(self, parent=None, caption="", directory="", filter=""):
        super(QFileDialogRestrictedToDirectory, self).__init__()
        self.setViewMode(QFileDialog.List)
        self.setAcceptMode(QFileDialog.AcceptOpen)
        self.RestrictedToDirectory = directory
        self.directoryEntered.connect(self.onDirectoryEntered) # here we restrict to select only data files in self.DefaultDirectory
    
    def onDirectoryEntered(self, dirpath):
        logger.debug('QFileDialogRestrictedToDirectory::onDirectoryEntered("{}")'.format(dirpath))
        self.setDirectory(self.RestrictedToDirectory)
        #<TODO># not working!
    
    def setRestrictedToDirectory(self, dirpath):
        self.RestrictedToDirectory = dirpath
        
    




class QLineEditForParamInput(QWidget):
    """ Not used. """
    
    ParamUpdateSignal = pyqtSignal(str, str, type, type)
    
    def __init__(self, labeltext, tooltiptext=None, datatype=str, listtype=str, parent=None):
        # datatype should be str, list, int, float
        # listtype should be list or str (stands for None)
        super(QLineEditForParamInput, self).__init__()
        self.ParamName = labeltext
        self.ParamDataType = datatype
        self.ParamListType = listtype
        self.ParamRegExpValidator = ''
        self.LineEdit = QLineEdit()
        #self.LineEdit.setMaximumWidth(120)
        self.LabelForLineEdit = QLabel(labeltext)
        self.LabelForLineEdit.setMinimumWidth(80)
        self.LayoutForLineEdit = QHBoxLayout()
        self.LayoutForLineEdit.setContentsMargins(0, 0, 0, 0)
        self.LayoutForLineEdit.setSpacing(0)
        self.LayoutForLineEdit.addWidget(self.LabelForLineEdit)
        self.LayoutForLineEdit.addWidget(self.LineEdit)
        self.setLayout(self.LayoutForLineEdit)
        if tooltiptext is not None:
            self.LineEdit.setToolTip(tooltiptext)
        if datatype is not None:
            if datatype is bool:
                self.ParamRegExpValidator = r'(True|False)'
            elif datatype is int:
                self.ParamRegExpValidator = r'([+-]|)[0-9.]+'
            elif datatype is float:
                self.ParamRegExpValidator = r'([+-]|)[0-9.]+([eE][0-9]+|[eE][+-][0-9]+)'
            else:
                pass
        if listtype is list:
            if self.ParamRegExpValidator == '':
                self.ParamRegExpValidator = r'\[.*\]'
            else:
                self.ParamRegExpValidator = r'\['+self.ParamRegExpValidator+r', *'+self.ParamRegExpValidator+r'.*'+r'\]'
        if self.ParamRegExpValidator != '':
            self.LineEdit.setValidator(QRegExpValidator(QRegExp(r'^'+self.ParamRegExpValidator+r'$')))
        self.LineEdit.setStyleSheet("""
            QLineEdit {
                background: white;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QLineEdit[readOnly="true"] {
                background: #dedede;
                border: 1px solid #cccccc;
            }
            """)
        self.LineEdit.textChanged.connect(self.onLineEditTextChangedCall)
    
    def text(self):
        return self.LineEdit.text()
    
    def toolTip(self):
        return self.LineEdit.toolTip()
    
    def setText(self, text):
        self.LineEdit.blockSignals(True)
        self.LineEdit.setText(str(text))
        self.LineEdit.blockSignals(False)

    def setEditable(self, editable):
        self.LineEdit.setReadOnly(operator.not_(editable))
        self.LineEdit.style().unpolish(self.LineEdit)
        self.LineEdit.style().polish(self.LineEdit)

    def setReadOnly(self, readonly):
        self.LineEdit.setReadOnly(readonly)
        self.LineEdit.style().unpolish(self.LineEdit)
        self.LineEdit.style().polish(self.LineEdit)

    @pyqtSlot(str)
    def onLineEditTextChangedCall(self, updatedtext):
        self.ParamUpdateSignal.emit(self.ParamName, updatedtext, self.ParamDataType, self.ParamListType)
        



class QCheckBoxForParamInput(QWidget):
    """ Not used. """
    
    ParamUpdateSignal = pyqtSignal(str, str, type, type)
    
    def __init__(self, labeltext, tooltiptext=None, datatype=bool, listtype=str, parent=None):
        super(QCheckBoxForParamInput, self).__init__()
        self.ParamName = labeltext
        self.ParamDataType = datatype
        self.ParamListType = listtype
        self.Layout = QHBoxLayout()
        self.Layout.setContentsMargins(0, 0, 0, 0)
        self.Layout.setSpacing(0)
        self.CheckBox = QCheckBox(labeltext)
        self.CheckBox.setMinimumWidth(160)
        #self.CheckBox.setMaximumWidth(160)
        self.Layout.addWidget(self.CheckBox)
        self.setLayout(self.Layout)
        if tooltiptext is not None:
            self.CheckBox.setToolTip(tooltiptext)
        #self.CheckBox.setStyleSheet("""
        #    QCheckBox {
        #        background: transparent;
        #        border: 1px solid #cccccc;
        #        border-radius: 3px;
        #    }
        #    """)
        self.CheckBox.stateChanged.connect(self.onCheckBoxStateChangedCall)
    
    def text(self):
        if self.CheckBox.isChecked():
            return 'True'
        else:
            return 'False'
    
    def toolTip(self):
        return self.CheckBox.toolTip()

    def setReadOnly(self, readonly):
        self.CheckBox.blockSignals(True)
        self.CheckBox.setReadOnly(readonly)
        self.CheckBox.blockSignals(False)

    def setChecked(self, checked):
        self.CheckBox.blockSignals(True)
        self.CheckBox.setChecked(checked)
        self.CheckBox.blockSignals(False)

    @pyqtSlot(int)
    def onCheckBoxStateChangedCall(self, state):
        if self.CheckBox.isChecked():
            self.ParamUpdateSignal.emit(self.ParamName, '1', self.ParamDataType, self.ParamListType)
        else:
            self.ParamUpdateSignal.emit(self.ParamName, '0', self.ParamDataType, self.ParamListType)



#
#
#
class QInputDialogSetLimits(QDialog):
    
    def __init__(self, vmin=None, vmax=None, title=None, parent=None):
        #logger.debug('QInputDialogSetLimits::__init__()')
        super(QInputDialogSetLimits, self).__init__(parent)
        layout = QFormLayout()
        self.Label1 = QLabel(self.tr('vmin'))
        self.Label2 = QLabel(self.tr('vmax'))
        self.LineEdit1 = QLineEdit('')
        self.LineEdit2 = QLineEdit('')
        self.LineEdit1.setMinimumWidth(200)
        self.LineEdit2.setMinimumWidth(200)
        self.LineEdit1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.LineEdit2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        if vmin is not None:
            self.LineEdit1.setText(str(vmin))
        if vmax is not None:
            self.LineEdit2.setText(str(vmax))
        self.ButtonBox = QDialogButtonBox()
        self.ButtonBox.addButton("OK", QDialogButtonBox.AcceptRole)
        self.ButtonBox.addButton("Cancel", QDialogButtonBox.RejectRole)
        #self.ButtonBox.setStandardButtons(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.ButtonBox.accepted.connect(self.accept)
        self.ButtonBox.rejected.connect(self.reject)
        layout.addRow(self.Label1, self.LineEdit1)
        layout.addRow(self.Label2, self.LineEdit2)
        layout.addRow(self.ButtonBox)
        self.LineEdit1.setValidator(QRegExpValidator(QRegExp(r'^([0-9eE .+-]+)$')))
        self.LineEdit2.setValidator(QRegExpValidator(QRegExp(r'^([0-9eE .+-]+)$')))
        self.setLayout(layout)
        if title is None:
            self.setWindowTitle(self.tr('Set Color Bar Limits'))
        else:
            self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.setMinimumHeight(100)
    
    def getResults(self):
        #logger.debug('QInputDialogSetLimits::getResults()')
        if self.exec_() == QDialog.Accepted:
            # get all values
            val1 = float(self.LineEdit1.text())
            val2 = float(self.LineEdit2.text())
            return val1, val2
        else:
            return None, None
    
    #def onAccepted(self):
    #    self.done(1)
    
    #def onRejected(self):
    #    self.done(0)




#
# class QFitsImageWidget
# for ImageA, ImageB, ImageC, ImageD, ImageE
#
class QFitsImageWidget(FigureCanvasQTAgg):
    
    PixelSelectedSignal = pyqtSignal(PixCoord)
    RegionSelectedSignal = pyqtSignal(PixelRegion)
    PositionVelocityRidgeSelectedSignal = pyqtSignal(list)
    
    class SelectingMode(Enum):
        Nothing = 0
        Pixel = 1
        Polygon = 2
        PositionVelocity = 3
    NoneMode = SelectingMode.Nothing
    PixelMode = SelectingMode.Pixel
    PolygonMode = SelectingMode.Polygon
    PositionVelocityMode = SelectingMode.PositionVelocity
    
    #def __colorbar_number_formatter__(self, x, pos):
    #    if x < 0:
    #        return '\N{MINUS SIGN}%g'%(-x)
    #    else:
    #        return '%g'%(x)
    
    def __init__(self, parent=None, width=5, height=5, title=None, name=None, with_colorbar=False, tight_layout=False):
        self.dataimage = None
        self.dataimagewcs = None
        self.dataimagewidth = None
        self.dataimageheight = None
        self.title = title
        self.name = name
        self.slit = None
        #self.pixmap = None
        #self.origin = QPoint(0, 0)
        #self.transform = None
        #self.scalefactor = None
        #self.datapixel = None # currently selected pixel in data coordinate (int)
        #self.datapolygon = None # polygons in data coordinate (int)
        #self.DataPixelF = None # currently selected pixel in data coordinate (float)
        #self.DataPolygonF = None # polygons in data coordinate (float)
        self.SelectedPixel = None # selected pixel in data coordinate (float)
        self.SelectedRegion = None # must be a PixelRegion
        self.SelectedPositionVelocityRidge = None # a list of positions for Position-Velocity (PV) analysis
        self.StoredRegions = []
        self.StoredPositionVelocityRidges = []
        self.SelectingMode = self.PixelMode
        self.PreviouslySelectedPixelCoord = None #
        self.PreviouslySelectedPixels = None # for adding pixels in polygon selection mode
        self.PlottedImshow = None
        self.PlottedContour = None
        self.PlottedColorbar = None
        self.PlottedPixel = [] # must be a list
        self.PlottedPolygon = [] # must be a list
        # setup matplotlib figure
        self.fig = Figure(figsize=(width, height), tight_layout=tight_layout)
        if not tight_layout:
            self.fig.subplots_adjust(left=0.01, right=0.80, bottom=0.01, top=0.87) #<TODO># adjust this
        self.axes = self.fig.add_subplot(111)
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)
        self.caxdiv = None
        self.cax = None
        self.caxorientation = "vertical"
        if with_colorbar:
            self.caxdiv = make_axes_locatable(self.axes)
            self.cax = self.caxdiv.append_axes("right", size="7%", pad="2%")
            self.cax.yaxis.set_ticks_position("right")
            self.cax.tick_params(labelsize='small')
            #self.cax.yaxis.set_major_formatter(ticker.FuncFormatter(self.__colorbar_number_formatter__))
            self.caxorientation = "vertical"
        #
        super(QFitsImageWidget, self).__init__(self.fig)
        self.mpl_connect('button_press_event', self.mplMousePressEvent)
        self.mpl_connect('motion_notify_event', self.mplMouseMoveEvent)
        self.mpl_connect('key_press_event', self.mplKeyPressEvent)
        # set qt cursor style
        self.setCursor(Qt.CrossCursor)
        #self.setStyleSheet('.QWidget{border: 2px solid black; border-radius: 2px; background-color: rgb(0, 0, 0);}')
        self.setMinimumSize(50, 50)
        #
        self.ContextMenu = QMenu()
        self.ActionSetColorBarLimits = QAction('Set Color Bar limits')
        self.ActionSetColorBarLimits.triggered.connect(self.onActionSetColorBarLimits)
        self.ContextMenu.addAction(self.ActionSetColorBarLimits)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.onImageViewerContextMenuCall)
        #
        if title is not None:
            self.axes.set_title(title)
    
    def setSelectingMode(self, selecting_mode):
        self.SelectingMode = selecting_mode
    
    def setImageWCS(self, data_image_wcs):
        self.dataimagewcs = data_image_wcs
    
    def showImage(self,
            image=None, cmap=None, extent=None, vmin=None, vmax=None,
            selected_pixel=None, selected_pixel_color='k', selected_pixel_alpha=0.9, selected_pixel_linewidth=1.5,
            selected_polygon=None, selected_polygon_color='k', selected_polygon_alpha=0.9, selected_polygon_linewidth=1.5,
            selected_polygon_closed=True,
            clear_plot=True, clear_selected_pixel=True, clear_selected_polygon=True,
            with_contour=False, with_colorbar=False, clear_colorbar=False,
            with_title=None, with_title_color='k',
            ):
        """
        selected_chunk are filled with color from zero baseline to y value.
        selected_channel is filled from ylim[0] to ylim[1].
        """
        # check axes
        if self.axes is None:
            return False
        # clear previous plot
        if clear_selected_pixel:
            for pobj in self.PlottedPixel:
                pobj.remove()
            self.PlottedPixel = []
        if clear_selected_polygon:
            for pobj in self.PlottedPolygon:
                pobj.remove()
            self.PlottedPolygon = []
        if clear_plot:
            for axitem in self.axes.lines + self.axes.collections + self.axes.texts:
                axitem.remove()
            if self.slit is not None:
                self.slit.remove()
                self.slit = None
        # set default cmap
        if cmap is None:
            #cmap = 'viridis' # 2022-08-25
            cmap = get_QFitsView_colormap_rainbow()
        # show image
        if image is not None:
            self.dataimage = image
            if extent is None:
                extent = [-0.5, image.shape[1]-0.5, -0.5, image.shape[0]-0.5] # left, right, bottom, top
            if self.PlottedImshow is None:
                imshow_params = {'origin':'lower', 'interpolation':'nearest', 'cmap':cmap, 'extent':extent}
                self.PlottedImshow = self.axes.imshow(self.dataimage, **imshow_params)
                #
                # fix axes range
                self.axes.set_xlim([extent[0], extent[1]])
                self.axes.set_ylim([extent[2], extent[3]])
                #
                if with_contour:
                    contour_params = {'origin':'lower', 'colors':'white', 'linewidths':0.605, 'alpha':0.85}
                    contour_params['extent'] = extent
                    self.PlottedContour = self.axes.contour(self.dataimage, **contour_params)
            else:
                self.PlottedImshow.set_data(self.dataimage)
                if vmin is None:
                    vmin = np.nanmin(self.dataimage)
                    #vmin = np.nanpercentile(self.dataimage.ravel(), 0.1)
                if vmax is None:
                    vmax = np.nanmax(self.dataimage)
                    #vmax = np.nanpercentile(self.dataimage.ravel(), 100.0)
                self.PlottedImshow.set_clim(vmin=vmin, vmax=vmax)
                self.PlottedImshow.set_extent(extent)
                #
                # fix axes range
                self.axes.set_xlim([extent[0], extent[1]])
                self.axes.set_ylim([extent[2], extent[3]])
                #
                if with_contour:
                    #if self.PlottedContour:
                    #    print('self.PlottedContour.collections', self.PlottedContour.collections)
                    #    for axitem in self.PlottedContour.collections:
                    #        print('axitem', axitem)
                    #        #axitem.remove()
                    #-- already removed by clear_plot
                    contour_params = {'origin':'lower', 'colors':'white', 'linewidths':0.605, 'alpha':0.85}
                    if extent is not None:
                        contour_params['extent'] = extent
                    self.PlottedContour = self.axes.contour(self.dataimage, **contour_params)
            #
            if with_colorbar:
                if clear_colorbar and self.PlottedColorbar is not None:
                    self.PlottedColorbar.remove()
                if self.PlottedColorbar is None:
                    if self.cax is not None:
                        self.PlottedColorbar = self.fig.colorbar(self.PlottedImshow, cax=self.cax, orientation=self.caxorientation)
                    else:
                        self.PlottedColorbar = self.fig.colorbar(self.PlottedImshow)
                #else:
                #    self.PlottedColorbar.set_clim(vmin=np.nanmin(self.dataimage), vmax=np.nanmax(self.dataimage))
                #    self.PlottedColorbar.draw_all()
            #
            if with_title is not None:
                self.axes.set_title(with_title, color=with_title_color)
                #self.axes.text(0.5, 1.02, with_title, ha='center', va='bottom', transform=self.axes.transAxes, color=with_title_color)
            #
            self.update()
        #
        # check image array
        if self.dataimage is None:
            return False
        #
        # highlight selected pixel
        if selected_pixel is not None:
            print('QFitsImageWidget::showImage()', 'selected_pixel', selected_pixel, 'value', self.dataimage[selected_pixel[1], selected_pixel[0]], 'widget name', self.name)
            pobj = Rectangle(
                [selected_pixel[0]-0.5, selected_pixel[1]-0.5],
                1, 1,
                linewidth=selected_pixel_linewidth,
                edgecolor=selected_pixel_color,
                facecolor='none',
                alpha=selected_pixel_alpha,
                zorder=99)
            self.axes.add_artist(pobj)
            self.PlottedPixel.append(pobj)
        #
        # highlight selected polygon
        if selected_polygon is not None:
            #print('QFitsImageWidget::showImage()', 'selected_polygon', selected_polygon)
            pobj = Polygon(
                selected_polygon,
                closed=selected_polygon_closed,
                linewidth=selected_polygon_linewidth,
                edgecolor=selected_polygon_color,
                facecolor='none',
                alpha=selected_polygon_alpha,
                zorder=99)
            self.axes.add_artist(pobj)
            self.PlottedPolygon.append(pobj)
        #
        # update canvas
        self.draw()
        return True
    
    def showSlit(self, slit_shape_in_pixel=None):
        if self.axes is not None and self.dataimage is not None:
            # if slit exists and not visible and slit_shape_in_pixel is None, set visible
            if self.slit is not None and self.slit.get_visible() == False and slit_shape_in_pixel is None:
                self.slit.set_visible(True)
            # otherwise if slit_shape_in_pixel is not None, plot the slit
            elif slit_shape_in_pixel is not None:
                #logger.debug('self.axes.get_xlim() = '+str(self.axes.get_xlim()))
                #logger.debug('self.axes.get_ylim() = '+str(self.axes.get_ylim()))
                #self.slit = self.axes.fill_between(**slit_shape_in_pixel, transform=self.axes.transData)
                #
                # slit in rectangle
                if self.slit is not None:
                    self.slit.remove()
                self.slit = slit_shape_in_pixel
                self.axes.add_patch(self.slit)
            else:
                return False
            self.update()
            self.draw()
            return True
        return False
    
    def hideSlit(self):
        if self.axes is not None and self.dataimage is not None and self.slit is not None:
            self.slit.set_visible(False)
            self.update()
            self.draw()
            return True
        return False
    
    def setSelectedPixel(self, x, y, do_plot=True, emit_signal=True):
        print('QFitsImageWidget::setSelectedPixel()', 'x: %s, y: %s, widget name: %s'%(x, y, self.name))
        if self.axes is not None and self.dataimage is not None:
            self.SelectedPixel = PixCoord(x=x, y=y)
            if do_plot:
                self.showImage(
                    clear_plot=False,
                    clear_selected_pixel=True,
                    clear_selected_polygon=False,
                    selected_pixel=(int(round(x)), int(round(y))))
            if emit_signal:
                self.PixelSelectedSignal.emit(self.SelectedPixel)
            return True
        return False
    
    def setSelectedRegion(self, region, closed=True, replaceId=-1):
        print('QFitsImageWidget::setSelectedRegion()', 'region: %s, closed: %s'%(str(region), closed))
        if self.axes is not None and self.dataimage is not None:
            if isinstance(region, PolygonPixelRegion):
                polygon = [(t.x,t.y) for t in region.vertices]
            elif isinstance(region, CirclePixelRegion):
                polygon = [(region.center.x+region.radius*np.cos(np.deg2rad(t)),
                            region.center.y+region.radius*np.sin(np.deg2rad(t))) for t in np.arange(0, 360+15, 15)]
                #<TODO># here we convert anything to polygon. will do better later.
            else:
                polygon = region # here we assume the input is like [(1,2), (3,4), (5,6), ...]
            self.showImage(
                clear_plot=False,
                clear_selected_pixel=False,
                clear_selected_polygon=True,
                selected_polygon=polygon,
                selected_polygon_closed=closed,
                with_contour=True,
                )
            if closed:
                if isinstance(region, PixelRegion):
                    self.SelectedRegion = region
                else:
                    x = [t[0] for t in polygon]
                    y = [t[1] for t in polygon]
                    self.SelectedRegion = PolygonPixelRegion(vertices=PixCoord(x=x, y=y))
                #patch = self.SelectedRegion.as_artist(facecolor='none', edgecolor='red', lw=2)
                #ax.add_patch(patch)
                if replaceId >= 0 and replaceId <= len(self.StoredRegions)-1:
                    self.StoredRegions[replaceId] = self.SelectedRegion
                elif self.SelectedRegion not in self.StoredRegions:
                    #self.StoredRegions.append(self.SelectedRegion) #<TODO># Store all drawn regions
                    self.StoredRegions = [self.SelectedRegion] #<TODO># Just store one region
                self.RegionSelectedSignal.emit(self.SelectedRegion)
            return True
        return False
    
    def unsetSelectedRegion(self):
        print('QFitsImageWidget::unsetSelectedRegion()')
        self.SelectedRegion = None
        return True
    
    def setSelectedPositionVelocityRidge(self, positions, closed=True, replaceId=-1):
        """
        'positions' should be a list of (x,y) pair.
        """
        print('QFitsImageWidget::setSelectedPositionVelocityRidge()', 'positions: %s, closed: %s'%(str(positions), closed))
        if self.axes is not None and self.dataimage is not None:
            self.showImage(
                clear_plot=False,
                clear_selected_pixel=False,
                clear_selected_polygon=True,
                selected_polygon=positions,
                selected_polygon_closed=False,
                with_contour=True,
                ) # when drawing pv ridge, it is always a non-closed polygon
            if closed:
                print('QFitsImageWidget::setSelectedPositionVelocityRidge()', 'self.SelectedPositionVelocityRidge = positions')
                self.SelectedPositionVelocityRidge = positions
                if replaceId >= 0 and replaceId <= len(self.StoredPositionVelocityRidges)-1:
                    self.StoredPositionVelocityRidges[replaceId] = self.SelectedPositionVelocityRidge
                elif self.SelectedPositionVelocityRidge not in self.StoredPositionVelocityRidges:
                    #self.StoredPositionVelocityRidges.append(self.SelectedPositionVelocityRidge) #<TODO># Store all drawn pv ridges
                    self.StoredPositionVelocityRidges = [self.SelectedPositionVelocityRidge] #<TODO># Just store one pv ridge
                self.PositionVelocityRidgeSelectedSignal.emit(self.SelectedPositionVelocityRidge)
            return True
        return False
    
    def unsetSelectedPositionVelocityRidge(self):
        print('QFitsImageWidget::unsetSelectedPositionVelocityRidge()')
        self.SelectedPositionVelocityRidge = None
        return True
    
    def mplMousePressEvent(self, event):
        if self.axes is not None and self.dataimage is not None:
            print('QFitsImageWidget::mplMousePressEvent()', 'event.button: %s, event.x: %s, event.y: %s, event.xdata:%s, event.ydata:%s'%(
                event.button, event.x, event.y, event.xdata, event.ydata))
            #print('QFitsImageWidget::mplMousePressEvent()', 'self.PreviouslySelectedPixels: %s'%(
            #    self.PreviouslySelectedPixels))
            #
            # if click in valid data area
            if event.xdata is not None and event.ydata is not None:
                # If in Polygon mode, right click once to enter the polygon editing mode, left click to add pixels,
                # then right click again to end the polygon editing mode.
                # Currently we can only add pixels in the editing mode.
                # If left click while not in the editing mode, then select a region from the store.
                if self.SelectingMode == self.PolygonMode:
                    if event.button == MouseButton.RIGHT:
                        if self.PreviouslySelectedPixels is None:
                            self.unsetSelectedRegion()
                            self.PreviouslySelectedPixels = [(event.xdata, event.ydata)]
                        else:
                            self.PreviouslySelectedPixels.extend([(event.xdata, event.ydata)]) # click right button again to close drawing the polygon
                            ok = self.setSelectedRegion(self.PreviouslySelectedPixels, closed=True)
                            self.PreviouslySelectedPixels = None
                    elif event.button == MouseButton.LEFT:
                        if self.PreviouslySelectedPixels is not None:
                            # if we are in polygon selection mode, then just add a pixel to the polygon and update the plot with unclosed polygon.
                            self.PreviouslySelectedPixels.extend([(event.xdata, event.ydata)])
                            ok = self.setSelectedRegion(self.PreviouslySelectedPixels, closed=False)
                        else:
                            # if we are not in polygon selection mode, i.e., self.PreviouslySelectedPixels is None,
                            # then left click still selects a pixel.
                            #ok = self.setSelectedPixel(event.xdata, event.ydata)
                            pix = PixCoord(x=event.xdata, y=event.ydata)
                            # then check if current position is in any region, if yes, then select that region.
                            self.unsetSelectedRegion()
                            if self.StoredRegions is not None:
                                for ireg in range(len(self.StoredRegions)-1, -1, -1):
                                    reg = self.StoredRegions[ireg] # from the last stored region to the first one
                                    print('self.StoredRegions[ireg=%d]: %s'%(ireg, str(reg)))
                                    if reg.contains(pix):
                                        ok = self.setSelectedRegion(reg, closed=True)
                                        break
                            self.PreviouslySelectedPixelCoord = pix
                #
                # If in PositionVelocity mode, right click once to enter the position editing mode, left click to add pixels,
                # then right click again to end the position editing mode.
                elif self.SelectingMode == self.PositionVelocityMode:
                    if event.button == MouseButton.RIGHT:
                        if self.PreviouslySelectedPixels is None:
                            self.unsetSelectedPositionVelocityRidge()
                            self.PreviouslySelectedPixels = [(event.xdata, event.ydata)] # TODO: how to discard the first right click position in the record?
                        else:
                            #self.PreviouslySelectedPixels.extend([(event.xdata, event.ydata)]) # click right button again to close drawing the position, but do not record the last right click position
                            ok = self.setSelectedPositionVelocityRidge(self.PreviouslySelectedPixels, closed=True)
                            self.PreviouslySelectedPixels = None
                    elif event.button == MouseButton.LEFT:
                        if self.PreviouslySelectedPixels is not None:
                            # if we are in position selection mode, then just add a pixel to the position and update the plot with unclosed position.
                            self.PreviouslySelectedPixels.extend([(event.xdata, event.ydata)])
                            ok = self.setSelectedPositionVelocityRidge(self.PreviouslySelectedPixels, closed=False)
                        else:
                            pass
                            ## if we are not in position selection mode, i.e., self.PreviouslySelectedPixels is None,
                            ## then left click still selects a pixel.
                            ##ok = self.setSelectedPixel(event.xdata, event.ydata)
                            #pix = PixCoord(x=event.xdata, y=event.ydata)
                            ## then check if current position is in any region, if yes, then select that region.
                            #self.unsetSelectedPositionVelocityRidge()
                            #if self.StoredPositionVelocityRidges is not None:
                            #    for ireg in range(len(self.StoredPositionVelocityRidges)-1, -1, -1):
                            #        reg = self.StoredPositionVelocityRidges[ireg] # from the last stored region to the first one
                            #        print('self.StoredPositionVelocityRidges[ireg=%d]: %s'%(ireg, str(reg)))
                            #        if reg.contains(pix):
                            #            ok = self.setSelectedPositionVelocityRidge(reg, closed=True)
                            #            break
                            #self.PreviouslySelectedPixelCoord = pix
                #
                # If in Pixel mode, then left click to select a pixel.
                elif self.SelectingMode == self.PixelMode:
                    if event.button == MouseButton.LEFT:
                        # if left click and in Pixel mode, select a pixel
                        ok = self.setSelectedPixel(event.xdata, event.ydata)
    
    def mplMouseMoveEvent(self, event):
        if self.axes is not None and self.dataimage is not None:
            #print('QFitsImageWidget::mplMouseMoveEvent()', 'event.button: %s, event.x: %s, event.y: %s, event.xdata:%s, event.ydata:%s'%(
            #    event.button, event.x, event.y, event.xdata, event.ydata))
            #
            # if click in valid data area
            if event.xdata is not None and event.ydata is not None:
                # if in Polygon mode
                if self.SelectingMode == self.PolygonMode:
                    # if currently mouse left button is pressed and moved, then move the polygon and update spectrum
                    if event.button == MouseButton.LEFT:
                        print('QFitsImageWidget::mplMouseMoveEvent()', 'event.button: %s, event.x: %s, event.y: %s, event.xdata:%s, event.ydata:%s'%(
                            event.button, event.x, event.y, event.xdata, event.ydata))
                        if self.SelectedRegion is not None and self.PreviouslySelectedPixelCoord is not None:
                            pix = PixCoord(x=event.xdata, y=event.ydata)
                            dx = pix.x-self.PreviouslySelectedPixelCoord.x
                            dy = pix.y-self.PreviouslySelectedPixelCoord.y
                            if isinstance(self.SelectedRegion, PolygonPixelRegion):
                                reg = PolygonPixelRegion(vertices=PixCoord(x=self.SelectedRegion.vertices.x+dx, y=self.SelectedRegion.vertices.y+dy))
                            elif isinstance(self.SelectedRegion, CirclePixelRegion):
                                reg = CirclePixelRegion(center=PixCoord(x=self.SelectedRegion.center.x+dx, y=self.SelectedRegion.center.y+dy),
                                                        radius=self.SelectedRegion.radius)
                            else:
                                raise NotImplementedError('Region type %s is not implemented!'%(type(self.SelectedRegion)))
                            ok = self.setSelectedRegion(reg, closed=True, replaceId=self.StoredRegions.index(self.SelectedRegion))
                            self.PreviouslySelectedPixelCoord = pix
    
    def mplKeyPressEvent(self, event):
        if self.axes is not None and self.dataimage is not None:
            print('QFitsImageWidget::mplKeyPressEvent()', 'event.key: %s, event.x: %s, event.y: %s, event.xdata:%s, event.ydata:%s'%(
                event.key, event.x, event.y, event.xdata, event.ydata))
            #
            # if click in valid data area
            if event.xdata is not None and event.ydata is not None:
                #
                # If in Pixel mode or Polygon mode, then press V key to print the coordinate of a pixel.
                if self.SelectingMode == self.PixelMode or self.SelectingMode == self.PolygonMode:
                    if event.key == 'v' or event.key == 'V':
                        # if press key 'v' or 'V' in Pixel mode, select a pixel and print its coordinate
                        ok = self.setSelectedPixel(event.xdata, event.ydata)
                        if self.dataimagewcs is not None:
                            try:
                                this_skycoord_RA, this_skycoord_Dec = self.dataimagewcs.wcs_pix2world([event.xdata], [event.ydata], 0) # pixel coord origin is 0,0
                                if not np.isscalar(this_skycoord_RA): this_skycoord_RA = this_skycoord_RA[0]
                                if not np.isscalar(this_skycoord_Dec): this_skycoord_Dec = this_skycoord_Dec[0]
                                print('QFitsImageWidget::mplKeyPressEvent()', 'event.key: %s, event.x: %s, event.y: %s, event.xdata:%s, event.ydata:%s, RA:%.10f, Dec:%.10f'%(
                                    event.key, event.x, event.y, event.xdata, event.ydata, this_skycoord_RA, this_skycoord_Dec))
                            except:
                                print('QFitsImageWidget::mplKeyPressEvent()', 'Error! No WCS is defined! Please call class function setImageWCS() first!')
    
    
    @pyqtSlot(QPoint)
    def onImageViewerContextMenuCall(self, point):
        self.ContextMenu.exec_(self.mapToGlobal(point))
    
    
    @pyqtSlot()
    def onActionSetColorBarLimits(self):
        logger.debug('QFitsImageWidget::onActionSetColorBarLimits()')
        if self.PlottedImshow is None:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText("No data image plotted.")
            msgBox.setWindowTitle("No data image plotted.")
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.exec()
            return
        else:
            vmin, vmax = self.PlottedImshow.get_clim()
            inputDialog = QInputDialogSetLimits(vmin, vmax)
            inputDialog.show()
            vmin, vmax = inputDialog.getResults()
            if vmin is not None and vmax is not None:
                self.PlottedImshow.set_clim(vmin=vmin, vmax=vmax)
                self.draw()
                self.update()
    
    
    #def setDataImage(self, DataImage):
    #    previous_data_image_width = self.dataimagewidth
    #    previous_data_image_height = self.dataimageheight
    #    self.dataimage = DataImage
    #    self.dataimagewidth = DataImage.shape[1]
    #    self.dataimageheight = DataImage.shape[0]
    #    self.datanorm = ImageNormalize(self.dataimage, interval=MinMaxInterval()) # , stretch=AsinhStretch()
    #    #self.rgb888image = np.zeros((self.dataimagewidth, self.dataimageheight, 3), dtype=np.uint8)
    #    #self.rgb888image[:,:,0] = (self.datanorm(self.dataimage).data*255).astype(np.uint8)
    #    self.uint8image = (self.datanorm(self.dataimage).data*255).astype(np.uint8)
    #    self.uint8image
    #    bytesPerLine = 1 * self.dataimagewidth
    #    self.qimage = QImage(self.uint8image, self.dataimagewidth, self.dataimageheight, bytesPerLine, QImage.Format_Grayscale8) # Format_Grayscale8 Format_RGB888 Format_RGBA8888 Format_RGB32
    #    self.qpixmap = QPixmap.fromImage(self.qimage)
    #    #self.transform = QTransform().scale(1, -1) # flip/mirror y so that image origin is at the lower left corner
    #    ok = self.setPixmap(self.qpixmap)
    #    if ok:
    #        if previous_data_image_width != self.dataimagewidth or previous_data_image_height != self.dataimageheight:
    #            # if image size changed, reset transform matrix
    #            ok = self.setDataTransform()
    #            if ok:
    #                self.update()
    #                return True
    #        else:
    #            self.update()
    #    return False
    
    #def setDataTransform(self):
    #    debug = 0
    #    if debug >= 1:
    #        print('QFitsImageWidget::setDataTransform()', 'self.size()', self.size())
    #    self.scalefactor = min(float(self.size().width())/self.dataimagewidth, float(self.size().height())/self.dataimageheight)
    #    if debug >= 1:
    #        print('QFitsImageWidget::setDataTransform()', 'self.scalefactor', self.scalefactor)
    #    self.transform = QTransform().scale(self.scalefactor, -self.scalefactor) # flip/mirror y so that image origin is at the lower left corner
    #    if debug >= 1:
    #        print('QFitsImageWidget::setDataTransform()', 'self.transform matrix: [[%+.3e %+.3e %+.3e]'%(self.transform.m11(), self.transform.m12(), self.transform.m13()))
    #        print('QFitsImageWidget::setDataTransform()', '                        [%+.3e %+.3e %+.3e]'%(self.transform.m21(), self.transform.m22(), self.transform.m23()))
    #        print('QFitsImageWidget::setDataTransform()', '                        [%+.3e %+.3e %+.3e]'%(self.transform.m31(), self.transform.m32(), self.transform.m33()))
    #    self.transform = QTransform().scale(self.scalefactor, -self.scalefactor).translate(0, -self.dataimageheight+1.) # flip/mirror y so that image origin is at the lower left corner
    #    if debug >= 1:
    #        print('QFitsImageWidget::setDataTransform()', 'self.transform matrix: [[%+.3e %+.3e %+.3e]'%(self.transform.m11(), self.transform.m12(), self.transform.m13()))
    #        print('QFitsImageWidget::setDataTransform()', '                        [%+.3e %+.3e %+.3e]'%(self.transform.m21(), self.transform.m22(), self.transform.m23()))
    #        print('QFitsImageWidget::setDataTransform()', '                        [%+.3e %+.3e %+.3e]'%(self.transform.m31(), self.transform.m32(), self.transform.m33()))
    #    return True
    
    #def setPixmap(self, pixmap):
    #    self.pixmap = pixmap
    #    return True
    
    #def setDataPixel(self, x, y):
    #    if self.scalefactor is not None and self.dataimagewidth is not None and self.dataimageheight is not None:
    #        ix = int(round(x))
    #        iy = int(round(y))
    #        if ix>=0 and ix<=self.dataimagewidth-1 and iy>=0 and iy<=self.dataimageheight-1:
    #            datapixel = QPoint(ix, iy)
    #            if self.datapixel != datapixel:
    #                self.datapixel = QPoint(ix, iy)
    #                self.DataPixelF = QPointF(float(x), float(y))
    #                self.update()
    #                self.PixelSelectedSignal.emit(ix, iy)
    #                return True
    #    return False
    
    #def setDataPixelFromEventPos(self, pos, printpixelvalue=False):
    #    if self.scalefactor is not None and self.dataimagewidth is not None and self.dataimageheight is not None:
    #        # get clicked pixel coordinate in data image pixel coordinate system
    #        x = (pos.x()-self.origin.x())
    #        y = (pos.y()-self.origin.y())
    #        x = x/self.scalefactor - 0.5 # pixel coordinates starting from (0,0), i.e., the first pixel's center has (0,0) and bottom-left corner (-0.5,-0.5)
    #        y = self.dataimageheight - (y/self.scalefactor) - 0.5 # mirrow y; pixel coordinates starting from (0,0), i.e., the first pixel's center has (0,0) and bottom-left corner (-0.5,-0.5)
    #        ok = self.setDataPixel(x, y)
    #        if ok:
    #            return True
    #    return False
    
    #def resizeEvent(self, event):
    #    if self.pixmap is not None:
    #        self.setDataTransform()
    #    super(QLabel, self).resizeEvent(event)
    
    #def paintEvent(self, event):
    #    #print('QFitsImageWidget::paintEvent()')
    #    if self.pixmap is not None and self.transform is not None:
    #        #print('QFitsImageWidget::paintEvent()', 'self.size()', self.size())
    #        size = self.size()
    #        painter = QPainter(self)
    #        #scaledPixmap = self.pixmap.scaled(size, Qt.KeepAspectRatio, transformMode=Qt.FastTransformation)
    #        scaledPixmap = self.pixmap.transformed(self.transform, Qt.FastTransformation)
    #        self.origin = QPoint(0, 0)
    #        self.origin.setX((size.width() - scaledPixmap.width()) / 2)
    #        self.origin.setY((size.height() - scaledPixmap.height()) / 2)
    #        painter.drawPixmap(self.origin, scaledPixmap)
    #        #
    #        if self.datapixel is not None:
    #            rect = QRectF(self.transform.map(self.datapixel)+self.origin, QSizeF(self.scalefactor,self.scalefactor))
    #            #print('QFitsImageWidget::paintEvent()', 'self.datapixel', self.datapixel, 'rect', rect)
    #            pen = QPen(Qt.green, 2, Qt.SolidLine)
    #            painter.setPen(pen)
    #            painter.drawRect(rect)
    #        #
    #    super(QLabel, self).paintEvent(event)
    
    #def mousePressEvent(self, event):
    #    #print('QFitsImageWidget::mousePressEvent()')
    #    #print('QFitsImageWidget::mousePressEvent()', 'event.pos()', event.pos(), 'event.buttons()', event.buttons())
    #    #print('QFitsImageWidget::mousePressEvent()', 'self.origin', self.origin)
    #    #
    #    if self.scalefactor is not None:
    #        # select pixel to plot spectrum if left click
    #        if event.buttons() == Qt.LeftButton :
    #            #print('QFitsImageWidget::mousePressEvent()', 'Left clicked')
    #            ok = self.setDataPixelFromEventPos(event.pos())
    #            #
    #            # also print pixel value
    #            #pixval = np.nan
    #            #if ok: pixval = self.dataimage[self.datapixel.y(), self.datapixel.x()]
    #            #print('QFitsImageWidget::mousePressEvent()', 'scalefactor: %s, event.pos(): %s, data pixel: %s (%s), pixel value: %s'%(\
    #            #    self.scalefactor, event.pos(), self.DataPixelF, self.datapixel, pixval))
    #        #
    #        if event.buttons() == Qt.RightButton :
    #            #print('QFitsImageWidget::mousePressEvent()', 'Right clicked')
    #            #if QApplication.keyboardModifiers() == Qt.ControlModifier:
    #            #    print('Ctrl')
    #            pass
    
    #def mouseMoveEvent(self, event):
    #    #
    #    if self.scalefactor is not None:
    #        # select pixel to plot spectrum
    #        if event.buttons() == Qt.LeftButton :
    #            #print('QFitsImageWidget::mouseMoveEvent()', 'Left pressed')
    #            ok = self.setDataPixelFromEventPos(event.pos())
    #        if event.buttons() == Qt.RightButton :
    #            #print('QFitsImageWidget::mouseMoveEvent()', 'Right pressed')
    #            pass




class QSpectrumWidget(FigureCanvasQTAgg):
    
    ChannelSelectedSignal = pyqtSignal(int)
    ChunkSelectedSignal = pyqtSignal(list)
    
    class SelectingMode(Enum):
        Channel = 1
        Chunk = 2
    ChannelMode = SelectingMode.Channel
    ChunkMode = SelectingMode.Chunk
    
    def __init__(self, parent=None, width=12, height=5, title=None, name=None, tight_layout=False):
        self.title = title
        self.name = name
        self.xarray = None
        self.yarray = None
        self.channelwidth = None
        self.SelectingMode = self.ChannelMode
        self.PreviouslySelectedChannel = None
        self.PlottedChannel = [] # must be a list
        self.PlottedChunk = [] # must be a list
        self.HasLegend = False
        # setup matplotlib figure
        self.fig = Figure(figsize=(width, height), tight_layout=tight_layout)
        self.axes = self.fig.add_subplot(111)
        super(QSpectrumWidget, self).__init__(self.fig)
        #self.axes.plot(np.arange(100), np.random.random(100)-0.5, drawstyle='steps-mid') # random initial spectrum
        #self.axes.set_xlabel('Channel')
        #self.axes.set_xticks([])
        #self.axes.set_yticks([])
        self.axes.tick_params(labelsize='small')
        self.mpl_connect('button_press_event', self.mplMousePressEvent)
        # set qt cursor style
        self.setCursor(Qt.CrossCursor)
        self.setStyleSheet('.QWidget{border: 2px solid black; border-radius: 2px; background-color: rgb(255, 255, 255);}')
        self.setMinimumSize(100, 100)
        #
        self.ContextMenu = QMenu()
        self.ActionSetXLimits = QAction('Set X limits')
        self.ActionSetYLimits = QAction('Set Y limits')
        self.ActionSetXLimits.triggered.connect(self.onActionSetXLimits)
        self.ActionSetYLimits.triggered.connect(self.onActionSetYLimits)
        self.ContextMenu.addAction(self.ActionSetXLimits)
        self.ContextMenu.addAction(self.ActionSetYLimits)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.onSpecViewerContextMenuCall)
        #
        if not tight_layout:
            #self.fig.subplots_adjust(left=0.18, right=0.98, bottom=0.13, top=0.87) #<TODO># adjust this
            self.fig.subplots_adjust(left=0.01, right=0.80, bottom=0.13, top=0.87) #<TODO># adjust this
            self.axes.yaxis.tick_right()
        #
        if title is not None:
            self.axes.set_title(title)
    
    def setSelectingMode(self, selecting_mode):
        self.SelectingMode = selecting_mode
    
    def determinePlotLimits(self, xarray, yarray, xmargin=0.1, ymargin=0.1):
        xmin = np.nanmin(xarray)
        xmax = np.nanmax(xarray)
        xdiv = xmax-xmin
        ymin = np.nanmin(yarray)
        ymax = np.nanmax(yarray)
        ydiv = ymax-ymin
        #print('QSpectrumWidget::determinePlotLimits()', 'xlim', [xmin-xmargin*xdiv, xmax+xmargin*xdiv], 'ylim', [ymin-ymargin*ydiv, ymax+ymargin*ydiv])
        return [xmin-xmargin*xdiv, xmax+xmargin*xdiv], [ymin-ymargin*ydiv, ymax+ymargin*ydiv]
    
    def plotSpectrum(self,
            x=None, y=None,
            xerr=None, yerr=None,
            xtype=None, ytype=None,
            xunit=None, yunit=None,
            color='C0', drawstyle='steps-mid', linestyle='solid', linewidth=1.0, label='__none__',
            plot_zero_line=True, update_xylimits=True,
            selected_chunk=None, selected_chunk_baseline=0.0, selected_chunk_color='yellow', selected_chunk_alpha=0.8,
            selected_channel=None, selected_channel_color='cyan', selected_channel_alpha=0.8,
            clear_plot=True, clear_text=True, clear_selected_chunk=True, clear_selected_channel=True,
            **kwargs,
            ):
        """
        selected_chunk are filled with color from zero baseline to y value.
        selected_channel is filled from ylim[0] to ylim[1].
        """
        # check axes
        if self.axes is None:
            return False
        # clear previous plot
        if clear_selected_channel:
            for pobj in self.PlottedChannel:
                pobj.remove()
            self.PlottedChannel = []
        if clear_selected_chunk:
            for pobj in self.PlottedChunk:
                pobj.remove()
            self.PlottedChunk = []
        if clear_text:
            for axitem in self.axes.texts:
                axitem.remove()
        if clear_plot:
            for axitem in self.axes.lines + self.axes.collections + self.axes.containers + self.axes.texts:
                if not clear_selected_channel and axitem in self.PlottedChannel:
                    continue
                elif not clear_selected_chunk and axitem in self.PlottedChunk:
                    continue
                elif not clear_text and axitem in self.axes.texts:
                    continue
                try:
                    axitem.remove()
                except:
                    pass
        # plot spectrum
        if x is not None and y is not None:
            self.xarray = np.array(x)
            self.yarray = np.array(y)
            if np.all(np.isnan(self.xarray)):
                return False
            if np.all(np.isnan(self.yarray)):
                return False
            self.channelwidth = np.abs(self.xarray[-1]-self.xarray[0]) / float(len(self.xarray)-1)
            # determine plotting limits
            if update_xylimits:
                xlim, ylim = self.determinePlotLimits(x, y)
                self.axes.set_xlim(xlim)
                self.axes.set_ylim(ylim)
            # plot x y
            if xerr is not None or yerr is not None:
                self.axes.errorbar(x, y, xerr=xerr, yerr=yerr, color=color, linestyle=linestyle, linewidth=linewidth, label=label, **kwargs)
            elif linestyle == 'none':
                self.axes.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label, **kwargs)
            else:
                self.axes.plot(x, y, color=color, drawstyle=drawstyle, linewidth=linewidth, label=label, **kwargs)
            # plot zero line
            if plot_zero_line:
                self.axes.plot(xlim, [0.0, 0.0], ls='dotted', lw=0.5, color='k')
        #
        # check x y arrays
        if self.xarray is None or self.yarray is None:
            return False
        #
        # highlight channel chunks
        # ('selected_chunk' contains channel index starting from 0)
        # ('selected_chunk' should contain pairs of channels)
        if selected_chunk is not None:
            self.PlottedChunk = []
            if np.isscalar(selected_chunk):
                ichan = selected_chunk
                selected_chunk = [(ichan,ichan)]
            for ichanpair in selected_chunk:
                if np.isscalar(ichanpair):
                    ichan = ichanpair
                    ichanpair = (ichan, ichan)
                else:
                    ichanpair = tuple(ichanpair)
                if len(ichanpair) >= 2:
                    ichan1 = min(ichanpair)
                    ichan2 = max(ichanpair)
                    if ichan1 < 0: ichan1 = 0
                    if ichan2 < 0: ichan2 = 0
                    if ichan1 > len(self.xarray)-1: ichan1 = len(self.xarray)-1
                    if ichan2 > len(self.xarray)-1: ichan2 = len(self.xarray)-1
                    highlighting_xarray = self.xarray[ichan1:ichan2+1]
                    highlighting_yarray = self.yarray[ichan1:ichan2+1]
                    hbar = self.axes.bar(
                        highlighting_xarray,
                        highlighting_yarray,
                        width=self.channelwidth,
                        align='center',
                        edgecolor='none',
                        facecolor=selected_chunk_color,
                        alpha=selected_chunk_alpha)
                    self.PlottedChunk.append(hbar)
        #
        # highlight current channel, fill between ylim[0] to ylim[1]
        if selected_channel is not None:
            ichan = int(round(selected_channel))
            if ichan < 0: ichan = 0
            if ichan > len(self.xarray)-1: ichan = len(self.xarray)-1
            pobj = self.axes.fill_between(
                [self.xarray[ichan]-self.channelwidth/2., self.xarray[ichan]+self.channelwidth/2.],
                [self.axes.get_ylim()[0], self.axes.get_ylim()[0]],
                [self.axes.get_ylim()[1], self.axes.get_ylim()[1]],
                color=selected_channel_color,
                alpha=selected_channel_alpha)
            self.PlottedChannel.append(pobj)
        #
        # plot axis label
        xlabel = ''
        if xtype is not None:
            xlabel = xtype
        if xunit is not None:
            xlabel = (xlabel+' '+'['+xunit+']').strip()
        if xlabel != '':
            self.axes.set_xlabel(xlabel)
        ylabel = ''
        if ytype is not None:
            ylabel = ytype
        if yunit is not None:
            ylabel = (ylabel+' '+'['+yunit+']').strip()
        if ylabel != '':
            self.axes.set_ylabel(ylabel)
        #
        # update canvas
        self.draw()
        return True
    
    def plotLegend(self, loc='upper left'):
        if self.axes:
            if not self.HasLegend:
                self.axes.legend(loc=loc)
                self.HasLegend = True
                # update canvas
                self.draw()
            return True
        return False
    
    def setSelectedChannel(self, ichan):
        if self.axes is not None and self.xarray is not None and self.yarray is not None:
            # highlight selected channel
            self.plotSpectrum(selected_channel=ichan, clear_plot=False, clear_selected_chunk=False, clear_selected_channel=True)
            #self.update()
            self.ChannelSelectedSignal.emit(ichan)
            return True
        return False
    
    def setSelectedChunk(self, chunk):
        if self.axes is not None and self.xarray is not None and self.yarray is not None:
            # highlight selected channel chunk
            self.plotSpectrum(selected_chunk=chunk, clear_plot=False, clear_selected_chunk=True, clear_selected_channel=False)
            #self.update()
            self.ChunkSelectedSignal.emit(chunk)
            # also emit channel selected signal
            ichanpair = chunk[0]
            ichanleft = min(ichanpair)
            ichanright = max(ichanpair)
            ichan = int(round((ichanleft+ichanright)/2.0))
            self.ChannelSelectedSignal.emit(ichan)
            return True
        return False
    
    def mplMousePressEvent(self, event):
        if self.axes is not None and self.xarray is not None and self.yarray is not None:
            print('QSpectrumWidget::mplMousePressEvent()', 'event.button: %s, event.x: %s, event.y: %s, event.xdata:%s, event.ydata:%s'%(
                event.button, event.x, event.y, event.xdata, event.ydata))
            # if click in valid data area
            if event.xdata is not None and event.ydata is not None:
                # if left click
                if event.button == MouseButton.LEFT:
                    ichan = np.argmin(np.abs(self.xarray-event.xdata)).ravel()[0]
                    if self.SelectingMode == self.ChannelMode:
                        ok = self.setSelectedChannel(ichan)
                        return ok
                    elif self.SelectingMode == self.ChunkMode:
                        if self.PreviouslySelectedChannel is None:
                            self.PreviouslySelectedChannel = ichan
                        else:
                            ok = self.setSelectedChunk([(self.PreviouslySelectedChannel, ichan)])
                            if ok:
                                self.PreviouslySelectedChannel = None
    
    #def setSelectedChannelFromEventPos(self, pos):
    #    if self.axes is not None and self.xarray is not None and self.yarray is not None:
    #        # translate screen coordinate to data coordinate
    #        x, y = self.axes.transData.inverted().transform((pos.x(), pos.y()))
    #        print('QSpectrumWidget::setSelectedChannelFromEventPos()', 'pos: %s, x: %s, y:%s'%(pos, x, y))
    #        #ok = self.setSelectedChannel(ichan)
    #    return False
    
    #def mousePressEvent(self, event):
    #    #print('QSpectrumWidget::mousePressEvent()')
    #    #print('QSpectrumWidget::mousePressEvent()', 'event.pos()', event.pos(), 'event.buttons()', event.buttons())
    #    #
    #    if self.axes is not None:
    #        # hightlight channel if left click
    #        if event.buttons() == Qt.LeftButton :
    #            #print('QSpectrumWidget::mousePressEvent()', 'Left clicked')
    #            #self.setSelectedChannelFromEventPos(event.pos())
    #        #
    #        if event.buttons() == Qt.RightButton :
    #            #print('QSpectrumWidget::mousePressEvent()', 'Right clicked')
    #            pass
    
    
    @pyqtSlot(QPoint)
    def onSpecViewerContextMenuCall(self, point):
        self.ContextMenu.exec_(self.mapToGlobal(point))
    
    
    @pyqtSlot()
    def onActionSetXLimits(self):
        logger.debug('QSpectrumWidget::onActionSetXLimits()')
        vmin, vmax = self.axes.get_xlim()
        inputDialog = QInputDialogSetLimits(vmin, vmax, title = self.tr("Set x-axis limits"))
        inputDialog.show()
        vmin, vmax = inputDialog.getResults()
        if vmin is not None and vmax is not None:
            self.axes.set_xlim([vmin, vmax])
            self.draw()
            self.update()
    
    
    @pyqtSlot()
    def onActionSetYLimits(self):
        logger.debug('QSpectrumWidget::onActionSetYLimits()')
        vmin, vmax = self.axes.get_ylim()
        inputDialog = QInputDialogSetLimits(vmin, vmax, title = self.tr("Set y-axis limits"))
        inputDialog.show()
        vmin, vmax = inputDialog.getResults()
        if vmin is not None and vmax is not None:
            self.axes.set_ylim([vmin, vmax])
            self.draw()
            self.update()



# 
# DS9 Colormap SLS
# 
def get_QFitsView_colormap_rainbow(bad='#dedede'):
    cmaprgb = np.array([
    [0.00000, 0.00000, 0.16471], 
    [0.02745, 0.00000, 0.18431], 
    [0.05882, 0.00000, 0.20000], 
    [0.08627, 0.00000, 0.21961], 
    [0.11373, 0.00000, 0.23922], 
    [0.14510, 0.00000, 0.25882], 
    [0.17647, 0.00000, 0.27843], 
    [0.20392, 0.00000, 0.29804], 
    [0.23137, 0.00000, 0.31765], 
    [0.26275, 0.00000, 0.33725], 
    [0.29412, 0.00000, 0.35686], 
    [0.32157, 0.00000, 0.37647], 
    [0.35294, 0.00000, 0.39608], 
    [0.38039, 0.00000, 0.41569], 
    [0.41176, 0.00000, 0.43529], 
    [0.43922, 0.00000, 0.45490], 
    [0.47059, 0.00000, 0.47451], 
    [0.49804, 0.00000, 0.49412], 
    [0.52941, 0.00000, 0.51373], 
    [0.55686, 0.00000, 0.53725], 
    [0.58824, 0.00000, 0.55686], 
    [0.55686, 0.00000, 0.57647], 
    [0.52941, 0.00000, 0.59608], 
    [0.49804, 0.00000, 0.61569], 
    [0.47059, 0.00000, 0.63922], 
    [0.43922, 0.00000, 0.65882], 
    [0.41176, 0.00000, 0.67843], 
    [0.38039, 0.00000, 0.70196], 
    [0.35294, 0.00000, 0.72157], 
    [0.32157, 0.00000, 0.74118], 
    [0.29412, 0.00000, 0.76471], 
    [0.26275, 0.00000, 0.78431], 
    [0.23137, 0.00000, 0.80392], 
    [0.20392, 0.00000, 0.82745], 
    [0.17647, 0.00000, 0.84706], 
    [0.14510, 0.00000, 0.87059], 
    [0.11373, 0.00000, 0.89020], 
    [0.08627, 0.00000, 0.91373], 
    [0.05882, 0.00000, 0.93333], 
    [0.02745, 0.00000, 0.95686], 
    [0.00000, 0.00000, 0.97647], 
    [0.00000, 0.00000, 1.00000], 
    [0.00000, 0.02353, 0.97647], 
    [0.00000, 0.04706, 0.95686], 
    [0.00000, 0.06275, 0.93333], 
    [0.00000, 0.08235, 0.91373], 
    [0.00000, 0.09804, 0.89020], 
    [0.00000, 0.11373, 0.87059], 
    [0.00000, 0.12941, 0.84706], 
    [0.00000, 0.14118, 0.82745], 
    [0.00000, 0.15686, 0.80392], 
    [0.00000, 0.16863, 0.78431], 
    [0.00000, 0.18431, 0.76471], 
    [0.00000, 0.19608, 0.74118], 
    [0.00000, 0.21176, 0.72157], 
    [0.00000, 0.22353, 0.70196], 
    [0.00000, 0.23529, 0.67843], 
    [0.00000, 0.25098, 0.65882], 
    [0.00000, 0.26275, 0.63922], 
    [0.00000, 0.27451, 0.61569], 
    [0.00000, 0.28627, 0.59608], 
    [0.00000, 0.29804, 0.57647], 
    [0.00000, 0.30980, 0.55686], 
    [0.00000, 0.32157, 0.53725], 
    [0.00000, 0.33333, 0.51373], 
    [0.00000, 0.34510, 0.49412], 
    [0.00000, 0.35686, 0.47451], 
    [0.00000, 0.36863, 0.45490], 
    [0.00000, 0.38039, 0.43529], 
    [0.00000, 0.39216, 0.41569], 
    [0.00000, 0.40392, 0.39608], 
    [0.00000, 0.41176, 0.37647], 
    [0.00000, 0.42353, 0.35686], 
    [0.00000, 0.43529, 0.33725], 
    [0.00000, 0.44706, 0.31765], 
    [0.00000, 0.45882, 0.29804], 
    [0.00000, 0.46667, 0.27843], 
    [0.00000, 0.47843, 0.25882], 
    [0.00000, 0.49020, 0.23922], 
    [0.00000, 0.49804, 0.21961], 
    [0.00000, 0.50980, 0.20000], 
    [0.00000, 0.52157, 0.18431], 
    [0.00000, 0.52941, 0.16471], 
    [0.00000, 0.54118, 0.14510], 
    [0.00000, 0.55294, 0.12941], 
    [0.00000, 0.56078, 0.10980], 
    [0.00000, 0.57255, 0.09412], 
    [0.00000, 0.58431, 0.07451], 
    [0.00000, 0.59216, 0.05882], 
    [0.00000, 0.60392, 0.04314], 
    [0.00000, 0.61176, 0.02745], 
    [0.00000, 0.62353, 0.01176], 
    [0.00000, 0.63137, 0.00000], 
    [0.00000, 0.64314, 0.00000], 
    [0.00000, 0.65098, 0.00000], 
    [0.00000, 0.66275, 0.00000], 
    [0.00000, 0.67059, 0.00000], 
    [0.00000, 0.68235, 0.00000], 
    [0.00000, 0.69020, 0.00000], 
    [0.00000, 0.70196, 0.00000], 
    [0.00000, 0.70980, 0.00000], 
    [0.00000, 0.72157, 0.00000], 
    [0.00000, 0.72941, 0.00000], 
    [0.00000, 0.74118, 0.00000], 
    [0.00000, 0.74902, 0.00000], 
    [0.00000, 0.76078, 0.00000], 
    [0.00000, 0.76863, 0.00000], 
    [0.00000, 0.77647, 0.00000], 
    [0.00000, 0.78824, 0.00000], 
    [0.00000, 0.79608, 0.00000], 
    [0.00000, 0.80784, 0.00000], 
    [0.00000, 0.81569, 0.00000], 
    [0.00000, 0.82353, 0.00000], 
    [0.00000, 0.83529, 0.00000], 
    [0.00000, 0.84314, 0.00000], 
    [0.00000, 0.85490, 0.00000], 
    [0.00000, 0.86275, 0.00000], 
    [0.00000, 0.87059, 0.00000], 
    [0.00000, 0.88235, 0.00000], 
    [0.00000, 0.89020, 0.00000], 
    [0.00000, 0.89804, 0.00000], 
    [0.00000, 0.90980, 0.00000], 
    [0.00000, 0.91765, 0.00000], 
    [0.00000, 0.92549, 0.00000], 
    [0.00000, 0.93725, 0.00000], 
    [0.00000, 0.94510, 0.00000], 
    [0.00000, 0.95294, 0.00000], 
    [0.00000, 0.96078, 0.00000], 
    [0.00000, 0.97255, 0.00000], 
    [0.00000, 0.98039, 0.00000], 
    [0.00000, 0.98824, 0.00000], 
    [0.00000, 1.00000, 0.00000], 
    [0.00000, 0.98824, 0.00000], 
    [0.00000, 0.98039, 0.00000], 
    [0.00000, 0.97255, 0.00000], 
    [0.00000, 0.96078, 0.00000], 
    [0.00000, 0.95294, 0.00000], 
    [0.00000, 0.94510, 0.00000], 
    [0.00000, 0.93725, 0.00000], 
    [0.00000, 0.92549, 0.00000], 
    [0.00000, 0.91765, 0.00000], 
    [0.00000, 0.90980, 0.00000], 
    [0.00000, 0.89804, 0.00000], 
    [0.00000, 0.89020, 0.00000], 
    [0.00000, 0.88235, 0.00000], 
    [0.00000, 0.87059, 0.00000], 
    [0.00000, 0.86275, 0.00000], 
    [0.00000, 0.85490, 0.00000], 
    [0.00000, 0.84314, 0.00000], 
    [0.00000, 0.83529, 0.00000], 
    [0.00000, 0.82353, 0.00000], 
    [0.00000, 0.81569, 0.00000], 
    [0.00000, 0.80784, 0.00000], 
    [0.00000, 0.79608, 0.00000], 
    [0.00000, 0.78824, 0.00000], 
    [0.00000, 0.77647, 0.00000], 
    [0.00784, 0.76863, 0.00000], 
    [0.03529, 0.77647, 0.00000], 
    [0.06667, 0.78824, 0.00000], 
    [0.09804, 0.80000, 0.00000], 
    [0.12941, 0.81176, 0.00000], 
    [0.16471, 0.82745, 0.00000], 
    [0.20000, 0.84314, 0.00000], 
    [0.23529, 0.85882, 0.00000], 
    [0.26667, 0.87059, 0.00000], 
    [0.30588, 0.89020, 0.00000], 
    [0.34118, 0.90196, 0.00000], 
    [0.37647, 0.92157, 0.00000], 
    [0.41176, 0.93333, 0.00000], 
    [0.44706, 0.95294, 0.00000], 
    [0.48627, 0.96863, 0.00000], 
    [0.52157, 0.98824, 0.00000], 
    [0.56078, 1.00000, 0.00000], 
    [0.59608, 1.00000, 0.00000], 
    [0.63529, 1.00000, 0.00000], 
    [0.67059, 1.00000, 0.00000], 
    [0.70980, 1.00000, 0.00000], 
    [0.74902, 1.00000, 0.00000], 
    [0.78431, 1.00000, 0.00000], 
    [0.82353, 1.00000, 0.00000], 
    [0.85882, 1.00000, 0.00000], 
    [0.89804, 1.00000, 0.00000], 
    [0.93333, 1.00000, 0.00000], 
    [0.97647, 1.00000, 0.00000], 
    [1.00000, 1.00000, 0.00000], 
    [1.00000, 1.00000, 0.00000], 
    [1.00000, 1.00000, 0.00000], 
    [1.00000, 1.00000, 0.00000], 
    [1.00000, 1.00000, 0.00000], 
    [1.00000, 1.00000, 0.00000], 
    [1.00000, 1.00000, 0.00000], 
    [0.99608, 1.00000, 0.00000], 
    [0.98039, 1.00000, 0.00000], 
    [0.96078, 0.97647, 0.00000], 
    [0.94510, 0.93725, 0.00000], 
    [0.92549, 0.89804, 0.00000], 
    [0.90980, 0.85882, 0.00000], 
    [0.89412, 0.81961, 0.00000], 
    [0.87451, 0.78039, 0.00000], 
    [0.85882, 0.74118, 0.00000], 
    [0.83922, 0.70196, 0.00000], 
    [0.82353, 0.66275, 0.00000], 
    [0.80392, 0.62353, 0.00000], 
    [0.78824, 0.58431, 0.00000], 
    [0.76863, 0.54510, 0.00000], 
    [0.75686, 0.50980, 0.00000], 
    [0.74118, 0.46667, 0.00000], 
    [0.72549, 0.43137, 0.00000], 
    [0.70980, 0.39216, 0.00000], 
    [0.69412, 0.35294, 0.00000], 
    [0.68235, 0.31765, 0.00000], 
    [0.66275, 0.27451, 0.00000], 
    [0.65098, 0.23922, 0.00000], 
    [0.63529, 0.20000, 0.00000], 
    [0.62745, 0.16863, 0.00000], 
    [0.61569, 0.12941, 0.00000], 
    [0.60784, 0.09804, 0.00000], 
    [0.61961, 0.08235, 0.00000], 
    [0.62745, 0.06275, 0.00000], 
    [0.63922, 0.04706, 0.00000], 
    [0.64706, 0.02353, 0.00000], 
    [0.65882, 0.00000, 0.00000], 
    [0.66667, 0.00000, 0.00000], 
    [0.67843, 0.00000, 0.00000], 
    [0.68627, 0.00000, 0.00000], 
    [0.69804, 0.00000, 0.00000], 
    [0.70980, 0.00000, 0.00000], 
    [0.71765, 0.00000, 0.00000], 
    [0.72941, 0.00000, 0.00000], 
    [0.73725, 0.00000, 0.00000], 
    [0.74902, 0.00000, 0.00000], 
    [0.75686, 0.00000, 0.00000], 
    [0.76863, 0.00000, 0.00000], 
    [0.77647, 0.00000, 0.00000], 
    [0.78824, 0.00000, 0.00000], 
    [0.80000, 0.00784, 0.00784], 
    [0.80784, 0.02745, 0.02745], 
    [0.81961, 0.05098, 0.05098], 
    [0.82745, 0.08235, 0.08235], 
    [0.83922, 0.11373, 0.11373], 
    [0.84706, 0.14902, 0.14902], 
    [0.85882, 0.19216, 0.19216], 
    [0.86667, 0.23137, 0.23137], 
    [0.87843, 0.27843, 0.27843], 
    [0.88627, 0.32549, 0.32549], 
    [0.89804, 0.37647, 0.37647], 
    [0.90980, 0.43137, 0.43137], 
    [0.91765, 0.48627, 0.48627], 
    [0.92941, 0.54118, 0.54118], 
    [0.93725, 0.60000, 0.60000], 
    [0.94902, 0.66275, 0.66275], 
    [0.95686, 0.72549, 0.72549], 
    [0.96863, 0.79216, 0.79216], 
    [0.97647, 0.85882, 0.85882], 
    [0.98824, 0.92941, 0.92941], 
    [1.00000, 1.00000, 1.00000], 
    ]) # QFitsView_colormaps_rainbow.lut
    cmap = ListedColormap(colors=cmaprgb, name='QFitsView_rainbow')
    if bad is not None:
        cmap.set_bad(bad)
    return cmap









#
# MAIN
#
if __name__ == '__main__':
    
    # read user input
    #DataFile = sys.argv[1]
    input_param_file = ''
    input_debug_mode = False
    if len(sys.argv) > 1:
        iarg = 1
        while iarg < len(sys.argv):
            argstr = sys.argv[iarg]
            if argstr.startswith('-'):
                argstr = re.sub(r'^[-]+', r'-', argstr.lower())
                if argstr == '-debug':
                    input_debug_mode = True
                    print('input_debug_mode = %s'%(input_debug_mode))
            else:
                if input_param_file == '' and sys.argv[iarg].endswith('.params'):
                    input_param_file = sys.argv[iarg]
                    print('input_param_file = %r'%(input_param_file))
            iarg += 1
    # 
    if input_debug_mode:
        logger.setLevel(logging.DEBUG)
        galaxy.logger.setLevel(logging.DEBUG)
        lensing.logger.setLevel(logging.DEBUG)
        utils_least_chi_squares_1d_fitter.logger.setLevel(logging.DEBUG)
        if 'DysmalPy' in logging.root.manager.loggerDict:
            logging.getLogger('DysmalPy').setLevel(logging.DEBUG)
    # print('logging.getLevelName(logger.level)', logging.getLevelName(logger.level))
    # print('logging.getLevelName(logging.getLogger(__name__).level)', logging.getLevelName(logging.getLogger(__name__).level))
    
    #
    app = QApplication([sys.argv[0]])
    
    ScreenSize = app.primaryScreen().size()
    
    logger.debug('ScreenSize = ' + str(ScreenSize.width()) + ', ' + str(ScreenSize.height()))
    
    app.setWindowIcon(getIconDysmalPy())

    # manager = multiprocessing.Manager()
    # queue = manager.Queue()
    # # tower = QDysmalPyFittingTower(basequeue=queue)
    # starship = QDysmalPyFittingStarship(queue = queue)
    # starship.start()
    # starship.join()
    #
    # time.sleep(30)
    
    w = QDysmalPyGUI(ScreenSize = ScreenSize)
    #w.resize(250, 150)
    #w.move(300, 300)
    #w.setWindowTitle('Simple')
    w.show()
    
    # load user input param file
    if input_param_file != '':
        w.selectParamFile(input_param_file)
    
    
    # exec
    sys.exit(app.exec_())








