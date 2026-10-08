import pandas
import h5py
import os
import utility
from utility.constants import Constants
import math
import numpy as np


def convert_from_cm2_to_m2(cross_section):
        return cross_section / 1.e4

def convert_from_m2_to_cm2(cross_section):
        return cross_section * 1.e4

def convert_from_cm_to_m(length):
        return length / 1.e2


def convert_from_10_19_to_1(density):
    return density * 1.e19


def calculate_velocity_from_energy(energy, mass):
    constants = Constants()
    velocity = (2 * energy * constants.charge_electron / mass) ** 0.5
    return velocity


def distance(a, b, system='cartesian'):
    assert len(a) == len(b), 'Points are not given in the same dimensions.'
    assert isinstance(system, str), 'Not a valid coordinate system.'
    assert system in ['cartesian'], system + ' is not a supported coordinate system.'

    if system == 'cartesian':
        s = sum((a[i]-b[i])**2 for i in range(len(a)))
        return math.sqrt(s)


def unit_vector(a, b, system='cartesian'):
    if system == 'cartesian':
        s = distance(a, b, system=system)
        return np.asarray([(a[i]-b[i])/s for i in range(len(a))])


def cartesian_to_cylin(point):
    assert len(point) == 3
    r, z = math.sqrt(point[0]**2 + point[1]**2), point[2]
    phi = math.atan2(point[1], point[0])
    return np.asarray([r, z, phi])


def cylin_to_cartesian(point):
    assert len(point) == 3
    x, y, z = point[0]*math.cos(point[2]), point[0]*math.sin(point[2]), point[1]
    return np.asarray([x, y, z])


def convert_keV_to_eV(energy):
    return energy*1E3
