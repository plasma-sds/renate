from lxml import etree
from utility.getdata import GetData
from utility.exceptions import InputError
from crm_solver.beamlet import Beamlet
from imas_utility.idsprofiles import ProfilesIds
from imas_utility.idsequilibrium import EquilibriumIds
import numpy as np
import pandas
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import utility.convert as uc


class BeamletFromIds:
    def __init__(self, input_source='beamlet/test_imas.xml'):
        self.access_path = input_source

        self.read_imas_xml()
        self.machine = self.param.getroot().find('body').find('imas_machine').text
        self.user = self.param.getroot().find('body').find('imas_user').text
        self.shotnumber = int(self.param.getroot().find('body').find('imas_shotnumber').text)
        self.runnumber = int(self.param.getroot().find('body').find('imas_runnumber').text)
        self.timeslice = float(self.param.getroot().find('body').find('imas_timeslice').text)

        self.profile_source = self.param.getroot().find('body').find('profile_source').text
        self.load_imas_profiles()

        self.equilibrium_source = self.param.getroot().find('body').find('device_magnetic_geometry').text
        self.load_imas_equilibrium()

        self.get_beamlet_current(current=0.002)
        self.get_beamlet_energy(energy=60)
        self.get_beamlet_species(species='Li')

        self.beamlet_profile_configuration()

    def read_imas_xml(self):
        self.param = GetData(data_path_name=self.access_path).data
        assert isinstance(self.param, etree._ElementTree)
        print('ElementTree read to dictionary from: ' + self.access_path)

    def get_beamlet_energy(self, energy=None):
        if (energy is not None) and (isinstance(energy, int)):
            beamlet_energy = etree.SubElement(self.param.getroot().find('body'), 'beamlet_energy', unit='keV')
            beamlet_energy.text = str(energy)
        else:
            print('Further data exploitation development is in process. Currently beam energy to be added manually')
            raise InputError('Missing beam energy input.')

    def get_beamlet_current(self, current=None):
        if (current is not None) and (isinstance(current, float)):
            beamlet_current = etree.SubElement(self.param.getroot().find('body'), 'beamlet_current', unit='A')
            beamlet_current.text = str(current)
        else:
            print('Further data exploitation development is in process. Currently beam current to be added manually')
            raise InputError('Missing beam current input.')

    def get_beamlet_species(self, species=None):
        if (species is not None) and (isinstance(species, str)):
            beamlet_species = etree.SubElement(self.param.getroot().find('body'), 'beamlet_species', unit='')
            beamlet_species.text = str(species)
        else:
            print('Further data exploitation development is in process. Currently beam species to be added manually')
            raise InputError('Missing beam species input.')

    def get_beamlet_ends(self):
        start = [float(self.param.getroot().find('body').find('beamlet_start').find('x').text),
                 float(self.param.getroot().find('body').find('beamlet_start').find('y').text),
                 float(self.param.getroot().find('body').find('beamlet_start').find('z').text)]
        end = [float(self.param.getroot().find('body').find('beamlet_end').find('x').text),
               float(self.param.getroot().find('body').find('beamlet_end').find('y').text),
               float(self.param.getroot().find('body').find('beamlet_end').find('z').text)]
        return np.asarray(start), np.asarray(end)

    def load_imas_profiles(self):
        if self.profile_source == 'core_profiles':
            self.run_prof = ProfilesIds(self.shotnumber, self.runnumber, self.profile_source)
        else:
            raise NotImplementedError('There is no input protocol for data stored in ' +
                                      self.profile_source + ' IDS. '
                                      'The requested IDS does not exist or data fetch for it is not implemented.')

    def load_imas_equilibrium(self):
        if self.equilibrium_source == 'equilibrium':
            self.equilibrium = EquilibriumIds(self.shotnumber, self.runnumber)
        else:
            raise NotImplementedError('There is no input protocol for data stored in ' +
                                      self.equilibrium_source + ' IDS. '
                                      'The requested IDS does not exist or data fetch for it is not implemented.')

    def beamlet_profile_configuration(self):
        ids_density = self.run_prof.get_electron_density(self.timeslice)
        ids_electron_temperature = self.run_prof.get_electron_temperature(self.timeslice)
        ids_ion_temperature = self.run_prof.get_ion_temperature(self.timeslice)
        ids_grid = self.run_prof.get_grid_in_psi(self.timeslice) / self.run_prof.get_grid_in_psi(self.timeslice)[-1]

        f_density = interp1d(ids_grid, ids_density)
        f_ion_temp = interp1d(ids_grid, ids_ion_temperature)
        f_electron_temp = interp1d(ids_grid, ids_electron_temperature)

        resolution = int(self.param.getroot().find('body').find('beamlet_resolution').text)
        start, end = self.get_beamlet_ends()
        beamlet_gird = np.linspace(0, uc.distance(start, end), resolution)
        beamlet_flux = self.beamlet_grid_psi(beamlet_gird, start, uc.unit_vector(start, end))

        beamlet_density = np.concatenate((self.profile_extrapol(beamlet_flux[np.where(beamlet_flux > 1)[0]],
                                                                [1, ids_density[-1]], [1.2, 1e+17]),
                                          f_density(beamlet_flux[np.where(beamlet_flux <= 1)[0]])))
        
        beamlet_electron_temp = np.concatenate((self.profile_extrapol(beamlet_flux[np.where(beamlet_flux > 1)[0]],
                                                                      [1, ids_electron_temperature[-1]], [1.2, 10]),
                                                f_electron_temp(beamlet_flux[np.where(beamlet_flux <= 1)[0]])))
        
        beamlet_ion_temp = np.concatenate((self.profile_extrapol(beamlet_flux[np.where(beamlet_flux > 1)[0]],
                                                                 [1, ids_ion_temperature[-1]], [1.2, 5]),
                                           f_ion_temp(beamlet_flux[np.where(beamlet_flux <= 1)[0]])))
        
        self.profiles = pandas.DataFrame(data={'beamlet_density': np.reshape(beamlet_density,beamlet_density.shape[0]),
                                               'beamlet_electron_temp': np.reshape(beamlet_electron_temp,beamlet_electron_temp.shape[0]),
                                               'beamlet_grid': beamlet_gird,
                                               'beamlet_ion_temp': np.reshape(beamlet_ion_temp,beamlet_ion_temp.shape[0])})

    def beamlet_grid_psi(self, beamlet_gird, start, vector):
        normalized_flux = self.equilibrium.get_normalized_2d_flux(self.timeslice)
        r_flux, z_flux = self.equilibrium.get_2d_equilibrium_grid(self.timeslice)
        flux_function = interp2d(r_flux, z_flux, normalized_flux, kind='cubic')
        beamlet_flux = []
        for distance in beamlet_gird:
            point = uc.cartesian_to_cylin(start - distance*vector)
            beamlet_flux.append(flux_function(point[0], point[1]))
        return np.asarray(beamlet_flux)

    def profile_extrapol(self, vector, boundary_point, reference_point):
        b = np.log(boundary_point[1] / reference_point[1]) / (boundary_point[0] - reference_point[0])
        a = boundary_point[1] / np.exp(b * boundary_point[0])
        return a * np.exp(b * vector)
