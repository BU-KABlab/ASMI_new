"""Test indentation on metal to get the internal elastic modulus of the system"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from src.CNCController import CNCController
from src.ForceSensor import ForceSensor
from src.ForceMonitoring import simple_indentation_measurement
from src.Analysis import analyze_file