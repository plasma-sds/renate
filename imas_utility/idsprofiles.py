from imas_utility.idsinstance import ImasObject
from utility.exceptions import IdsInstanceLoadError
from utility.exceptions import IdsAttributeLoadError
import numpy as np


class ProfilesIds(ImasObject):
    def __init__(self, shot, run, source, user=None, machine=None):
        super().__init__(shot, run, user, machine)
        assert isinstance(source, str), 'Source must be of str type.'
        assert (source in ['core_profiles', 'edge_profiles']), 'Supported IDS sources for renate-od are:' + \
            'core_profiles and edge_profiles.'
        self.load_profiles_from_ids(source)

    def load_profiles_from_ids(self, source):
        self.source = source
        try:
            self.profiles = self.imas_pointer.get(self.source)
        except Exception as exc:
            raise IdsInstanceLoadError('No ' + self.source + ' IDS was found for shot: '
                                       + str(self.shot) + ' at run ' + str(self.run)) from exc

    def get_grid_in_rho_tor_norm(self, time):
        time_index = self.get_time_index(time)
        try:
            return self.profiles.profiles_1d[time_index].grid.rho_tor_norm
        except Exception as exc:
            raise IdsAttributeLoadError('There is no available rho tor norm based grid for Shot:' +
                                        str(self.shot) + ' at Run: ' + str(self.run) + ' in ' + self.source + ' IDS.') from exc

    def get_grid_in_psi(self, time):
        time_index = self.get_time_index(time)
        try:
            return self.profiles.profiles_1d[time_index].grid.psi
        except Exception as exc:
            raise IdsAttributeLoadError('There is no available psi based grid for Shot:' +
                                        str(self.shot) + ' at Run: ' + str(self.run) + ' in ' + self.source + ' IDS.') from exc

    def get_grid_in_rho_tor(self, time):
        time_index = self.get_time_index(time)
        try:
            return self.profiles.profiles_1d[time_index].grid.rho_tor
        except Exception as exc:
            raise IdsAttributeLoadError('There is no available rho tor based grid for Shot:' +
                                        str(self.shot) + ' at Run: ' + str(self.run) + ' in ' + self.source + ' IDS') from exc

    def get_electron_density(self, time):
        time_index = self.get_time_index(time)
        try:
            return self.profiles.profiles_1d[time_index].electrons.density
        except Exception as exc:
            raise IdsAttributeLoadError('There is no available electron density for Shot:' +
                                        str(self.shot) + ' at Run: ' + str(self.run) + ' in ' + self.source + ' IDS') from exc

    def get_electron_temperature(self, time):
        time_index = self.get_time_index(time)
        try:
            return self.profiles.profiles_1d[time_index].electrons.temperature
        except Exception as exc:
            raise IdsAttributeLoadError('There is no available electron temperature for Shot:' +
                                        str(self.shot) + ' at Run: ' + str(self.run) + ' in ' + self.source + ' IDS') from exc

    def get_ion_temperature(self, time):
        time_index = self.get_time_index(time)
        try:
            return self.profiles.profiles_1d[time_index].ion[1].temperature
        except Exception as exc:
            raise IdsAttributeLoadError('There is no available D temperature for Shot:' +
                                        str(self.shot) + ' at Run: ' + str(self.run) + ' in ' + self.source + ' IDS') from exc

    def get_time_index(self, time):
        try:
            time_array = self.profiles.time
        except Exception as exc:
            raise IdsAttributeLoadError('No time array available for the requested Shot: ' +
                                        str(self.shot) + ' and Run: ' + str(self.run) + ' in ' + self.source + ' IDS') from exc

        if (time_array[time_array.argmin()] <= time) and (time_array[time_array.argmax()] >= time):
            return (np.abs(time_array - time)).argmin()
        else:
            raise ValueError('Time value : ' + str(time) + ' is out of bound. Min time instance is ' +
                             str(time_array[time_array.argmin]) + '. Max time instance is : ' +
                             str(time_array[time_array.argmax]))
