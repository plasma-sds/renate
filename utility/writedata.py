import os
from utility.getdata import GetData
from lxml import etree


class WriteData:
    def __init__(self, root_path=None):
        if root_path is None:
            root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data") + os.sep
        self.root_path = root_path

    def write_beamlet_profiles(self, beamlet, subdir=''):
        output_path = beamlet.param.getroot().find('head').find('id').text
        h5_output_path = subdir + output_path + ".h5"
        xml_output_path = subdir + output_path + ".xml"
        GetData.ensure_dir(self.root_path + h5_output_path)
        try:
            beamlet.profiles.to_hdf(path_or_buf=self.root_path + h5_output_path, key="profiles")
            beamlet.components.to_hdf(path_or_buf=self.root_path + h5_output_path, key="components")
            if not isinstance(beamlet.param.getroot().find('body').find('beamlet_history'), etree._Element):
                new_element = etree.Element('beamlet_history')
                new_element.text = beamlet.param.getroot().find('body').find('beamlet_source').text
                new_element.set('unit', '-')
                beamlet.param.getroot().find('body').append(new_element)
            else:
                beamlet.param.getroot().find('body').find('beamlet_history').text = \
                    beamlet.param.getroot().find('body').find('beamlet_source').text
            beamlet.param.getroot().find('body').find('beamlet_source').text = h5_output_path
            beamlet.param.write(self.root_path + xml_output_path)
            print('Beamlet profile data written to file: ' + subdir + output_path)
            return self.root_path+h5_output_path, self.root_path+xml_output_path
        except:
            raise Exception('Beamlet profile data could NOT be written to file: ' + subdir + output_path)

    def write_photon_emission_profile(self, obs_param, emission_profiles, subdir=''):
        output_path = obs_param.getroot().find('head').find('id').text
        h5_output_path = self.root_path + subdir + output_path + ".h5"
        xml_output_path = self.root_path + subdir + output_path + ".xml"
        GetData.ensure_dir(h5_output_path)
        try:
            emission_profiles.to_hdf(path_or_buf=h5_output_path, key='emission_profiles')
            if not isinstance(obs_param.getroot().find('body').find('emission_profiles'), etree._Element):
                new_element = etree.Element('emission_profiles')
                new_element.text = h5_output_path
                new_element.set('unit', '-')
                obs_param.getroot().find('body').append(new_element)
            obs_param.write(xml_output_path)
            print('Photon emission profile data written to file: ' + output_path)
        except:
            raise Exception('Photon emission profile data could NOT be written to file: ' + output_path)
