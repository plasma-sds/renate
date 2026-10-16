import os
import h5py
import numpy
from utility.getdata import GetData, AccessData
from lxml import etree
from utility.accessdata import DEFAULT_SETUP, FALLBACK_SETUP

OUTPUT_SUBDIR = 'output'


class WriteData:
    def __init__(self, root_path=None):
        if root_path is None:
            access_data = AccessData(DEFAULT_SETUP)
            root_path = os.path.join(access_data.user_local_data_directory, OUTPUT_SUBDIR) + os.sep
        self.root_path = root_path

    def write_beamlet_profiles(self, beamlet, subdir=''):
        output_path = beamlet.param.getroot().find('head').find('id').text
        h5_output_path = os.path.join(subdir, output_path + ".h5")
        xml_output_path = os.path.join(subdir, output_path + ".xml")
        h5_full_path = os.path.join(self.root_path, h5_output_path)
        xml_full_path = os.path.join(self.root_path, xml_output_path)
        GetData.ensure_dir(h5_full_path)
        try:
            beamlet.profiles.to_hdf(path_or_buf=h5_full_path, key="profiles")
            beamlet.components.to_hdf(path_or_buf=h5_full_path, key="components")
            if not isinstance(beamlet.param.getroot().find('body').find('beamlet_history'), etree._Element):
                new_element = etree.Element('beamlet_history')
                new_element.text = beamlet.param.getroot().find('body').find('beamlet_source').text
                new_element.set('unit', '-')
                beamlet.param.getroot().find('body').append(new_element)
            else:
                beamlet.param.getroot().find('body').find('beamlet_history').text = \
                    beamlet.param.getroot().find('body').find('beamlet_source').text
            beamlet.param.getroot().find('body').find('beamlet_source').text = h5_output_path
            beamlet.param.write(xml_full_path)
            print('Beamlet profile data written to file: ' + os.path.join(subdir, output_path))
            return h5_full_path, xml_full_path
        except Exception as exc:
            raise OSError('Beamlet profile data could NOT be written to file: '
                          + os.path.join(subdir, output_path)) from exc

    def write_observation(self, obs, output_id, subdir=''):
        if not hasattr(obs, 'photon_flux'):
            raise ValueError('No photon flux is stored in the observation. '
                             'Call compute_photon_flux() with <store=True> first.')
        h5_full_path = os.path.join(self.root_path, subdir, output_id + '.h5')
        GetData.ensure_dir(h5_full_path)
        try:
            with h5py.File(h5_full_path, 'w') as h5_file:
                for name, data, unit in (('detector_positions', obs.detector_positions, 'm'),
                                         ('detector_size', obs.detector_size, 'm'),
                                         ('detector_ranges', obs.detector_ranges, 'm'),
                                         ('photon_flux', numpy.atleast_2d(obs.photon_flux), 'ph/s')):
                    h5_file.create_dataset(name, data=data).attrs['unit'] = unit
                h5_file.create_dataset('observed_transitions', data=obs.observed_transitions,
                                       dtype=h5py.string_dtype())
                h5_file.create_dataset('observed_beamlets', data=['' if source is None else source
                                                                  for source in obs.observed_beamlets],
                                       dtype=h5py.string_dtype())
            print('Observation data written to file: ' + os.path.join(subdir, output_id + '.h5'))
            return h5_full_path
        except Exception as exc:
            raise OSError('Observation data could NOT be written to file: '
                          + os.path.join(subdir, output_id + '.h5')) from exc
