import pandas
import h5py
import os
import utility
from utility.constants import Constants
import math
import numpy as np


def convert_beamlet_profiles(data_path_name='data/beamlet/test_profiles.h5'):
    """
    Convert HDF profiles file saved by RENATE to HDF file of the format of RENATE-OD.
    :param data_path_name: Old file is copied to .old, and then overwritten.
    :return: 
    """
    h5file = h5py.File(data_path_name, 'r')
    pandas_profiles = pandas.DataFrame()
    columns = list(h5file.keys())
    for column in columns:
        pandas_profiles[column] = h5file[column].value
    h5file.close()
    os.rename(os.path.join(os.path.dirname(__file__),'..', data_path_name),
              os.path.join(os.path.dirname(__file__),'..', data_path_name +'.old'))
    pandas_profiles.to_hdf(data_path_name)


def convert_from_cm2_to_m2(cross_section):
        return cross_section / 1.e4

def convert_from_m2_to_cm2(cross_section):
        return cross_section * 1.e4

def convert_from_cm_to_m(length):
        return length / 1.e2


def convert_from_10_19_to_1(density):
    return density * 1.e19


def convert_beamlet_profiles_to_si(data_path_name='beamlet/test_profiles.h5'):
    pandas_profiles = utility.getdata.GetData(data_path_name=data_path_name).data
    assert isinstance(pandas_profiles, pandas.DataFrame)
    full_data_path_name = os.path.join('data/', data_path_name)
    pandas_profiles['beamlet_density'] = convert_from_10_19_to_1(pandas_profiles['beamlet_density'])
    os.rename(os.path.join(os.path.dirname(__file__),'..', full_data_path_name),
              os.path.join(os.path.dirname(__file__), '..', full_data_path_name + '.non-si'))
    pandas_profiles.to_hdf(os.path.join(os.path.dirname(__file__),'..', full_data_path_name), 'profiles')
    print('Beamlet.param converted to SI in file: ' + full_data_path_name)


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
