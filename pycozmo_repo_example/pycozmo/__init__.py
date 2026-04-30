"""

PyCozmo - a pure-Python Cozmo robot communication library.

"""

import sys

from .logger import *
from .run import *

from .frame import Frame
from .conn import ROBOT_ADDR
from .client import Client
from .robot import *
from .event import *

from ...prev.pycozmo.pycozmo import exception
from ...prev.pycozmo.pycozmo import util
from ...prev.pycozmo.pycozmo import window
from ...prev.pycozmo.pycozmo import protocol_base
from ...prev.pycozmo.pycozmo import protocol_declaration
from ...prev.pycozmo.pycozmo import protocol_generator
from ...prev.pycozmo.pycozmo import protocol_encoder
from ...prev.pycozmo.pycozmo import protocol_utils
from ...prev.pycozmo.pycozmo import lights
from ...prev.pycozmo.pycozmo import camera
from ...prev.pycozmo.pycozmo import object
from ...prev.pycozmo.pycozmo import filter
from ...prev.pycozmo.pycozmo import anim
from ...prev.pycozmo.pycozmo import anim_encoder
from ...prev.pycozmo.pycozmo import image_encoder
from ...prev.pycozmo.pycozmo import procedural_face
from ...prev.pycozmo.pycozmo import activity
from ...prev.pycozmo.pycozmo import behavior
from ...prev.pycozmo.pycozmo import emotions
from ...prev.pycozmo.pycozmo import brain
from ...prev.pycozmo.pycozmo import audiokinetic
from ...prev.pycozmo.pycozmo import expressions


__version__ = "0.8.0"

__all__ = [
    "logger",
    "logger_protocol",
    "logger_robot",

    "Frame",
    "ROBOT_ADDR",
    "Client",

    "setup_basic_logging",
    "connect",
]

if sys.version_info < (3, 6, 0):
    sys.exit("ERROR: PyCozmo requires Python 3.6.0 or newer.")
del sys
