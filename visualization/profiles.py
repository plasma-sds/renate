import matplotlib.pyplot
import utility
from matplotlib.backends.backend_pdf import PdfPages
import datetime
from atomic.atomic_db import AtomicDB


class BeamletProfiles:
    def __init__(self, beamlet=None, param_path='output/beamlet/beamlet_test.xml'):
        if beamlet is None:
            self.param_path = param_path
            self.param = utility.getdata.GetData(data_path_name=self.param_path).data
            self.access_path = self.param.getroot().find('body').find('beamlet_source').text
            self.components = utility.getdata.GetData(data_path_name=self.access_path, data_key=['components']).data
            self.profiles = utility.getdata.GetData(data_path_name=self.access_path, data_key=['profiles']).data
            self.atomic_db = AtomicDB(param=self.param, components=self.components)
            self.title = None
        else:
            self.param = beamlet.param
            self.profiles = beamlet.profiles
            self.components = beamlet.components
            self.atomic_db = beamlet.atomic_db

    def set_x_range(self, x_min=None, x_max=None):
        self.x_limits = [x_min, x_max]

    def plot_RENATE_bechmark(self, plot_type='population'):
        fig1 = matplotlib.pyplot.figure()
        grid = matplotlib.pyplot.GridSpec(3, 1)
        ax1 = matplotlib.pyplot.subplot(grid[0, 0])
        ax1 = self.__setup_density_axis(ax1)
        ax2 = ax1.twinx()
        self.__setup_temperature_axis(ax2)
        self.title = 'Plasma profiles'
        ax1.set_title(self.title)
        self.__setup_RENATE_benchmark_axis(matplotlib.pyplot.subplot(grid[1:, 0]), plot_type)
        matplotlib.pyplot.show()

    def __setup_RENATE_benchmark_axis(self, axis, plot_type):
        max_val = self.profiles['level ' + self.atomic_db.inv_atomic_dict[0]][0]
        for level in self.atomic_db.atomic_dict.keys():
            if plot_type == 'population':
                axis.plot(self.profiles['beamlet grid'], self.profiles['RENATE level ' +
                          str(self.atomic_db.atomic_dict[level])], '-', label='RENATE '+level)
                axis.plot(self.profiles['beamlet grid'], self.profiles['level '+level]/max_val,
                          '--', label='ROD '+level)
                axis.set_ylabel('Relative electron population [-]')
                axis.set_yscale('log', nonpositive='clip')
            elif plot_type == 'error':
                axis.set_ylabel('Relative error [-]')
                axis.plot(self.profiles['beamlet grid'], abs(self.profiles['level '+level]/max_val -
                          self.profiles['RENATE level ' + str(self.atomic_db.atomic_dict[level])]) /
                          (self.profiles['level '+level]/max_val), '--', label='Level '+level)
            else:
                raise ValueError('Grid type ' + plot_type + 'not implemented!')
        if hasattr(self, 'x_limits'):
            axis.set_xlim(self.x_limits)
        axis.legend(loc='best', ncol=1)
        self.title = 'Benchmark: RENATE - ROD'
        axis.set_title(self.title)
        axis.grid()
        return axis

    def plot_linear_emission_density(self, from_level=None, to_level=None):
        axis_dens = matplotlib.pyplot.subplot()
        self.__setup_density_axis(axis_dens)
        axis_dens.set_xlabel('Distance [m]')
        axis_em = axis_dens.twinx()
        if from_level is None or to_level is None or not isinstance(from_level, str) or not isinstance(to_level, str):
            from_level, to_level, ground_level, transition = self.atomic_db.set_default_atomic_levels()
        else:
            transition = from_level + '-' + to_level
        self.__setup_linear_emission_density_axis(axis_em, transition)
        matplotlib.pyplot.show()

    def __setup_linear_emission_density_axis(self, axis, transition):
        try:
            axis.plot(self.profiles['beamlet grid'], self.profiles[transition],
                      label='Emission for '+transition, color='r')
        except KeyError:
            raise Exception('The requested transition: <'+transition+'> is not in the stored data. '
                            'Try computing it first or please make sure it exists')
        axis.set_ylabel('Linear emission density [ph/sm]')
        axis.yaxis.label.set_color('r')
        axis.legend(loc='upper right')
        return axis

    def plot_attenuation(self):
        axis_dens = matplotlib.pyplot.subplot()
        self.__setup_density_axis(axis_dens)
        axis_dens.set_xlabel('Distance [m]')
        axis_em = axis_dens.twinx()
        self.__setup_linear_density_attenuation_axis(axis_em)
        matplotlib.pyplot.show()

    def __setup_linear_density_attenuation_axis(self, axis):
        axis.plot(self.profiles['beamlet grid'], self.profiles['linear_density_attenuation'],
                  label='Linear density attenuation', color='r')
        axis.set_ylabel('Linear density [1/m]')
        axis.yaxis.label.set_color('r')
        axis.legend(loc='upper right')
        return axis

    def plot_relative_populations(self):
        axis = matplotlib.pyplot.subplot()
        self.__setup_population_axis(axis, kind='relative')
        matplotlib.pyplot.show()

    def plot_populations(self):
        axis = matplotlib.pyplot.subplot()
        self.__setup_population_axis(axis)
        matplotlib.pyplot.show()

    def plot_all_profiles(self):
        fig1 = matplotlib.pyplot.figure()
        grid = matplotlib.pyplot.GridSpec(3, 1)
        ax1 = matplotlib.pyplot.subplot(grid[0, 0])
        ax1 = self.__setup_density_axis(ax1)
        ax2 = ax1.twinx()
        self.__setup_temperature_axis(ax2)
        self.title = 'Plasma profiles'
        ax1.set_title(self.title)
        ax3 = matplotlib.pyplot.subplot(grid[1:, 0])
        self.__setup_population_axis(ax3)
        fig1.tight_layout()
        matplotlib.pyplot.show()

    def benchmark(self, benchmark_param_path='../data/beamlet/IMAS_beamlet_test_profiles_Li.xml', key=['profiles']):
        benchmark_param = utility.getdata.GetData(data_path_name=benchmark_param_path).data
        benchmark_path = benchmark_param.getroot().find('body').find('beamlet_profiles').text
        benchmark_profiles = utility.getdata.GetData(data_path_name=benchmark_path, data_key=key).data
        fig1 = matplotlib.pyplot.figure()
        ax1 = matplotlib.pyplot.subplot()
        ax1 = self.__setup_population_axis(ax1)
        ax1 = self.setup_benchmark_axis(benchmark_profiles, axis=ax1)
        ax1.legend(loc='best', ncol=2)
        self.title = 'Beamlet profiles - benchmark'
        ax1.set_title(self.title)
        ax1.grid()
        fig1.tight_layout()
        matplotlib.pyplot.show()

    def __setup_density_axis(self, axis):
        for comp in self.components.T.keys():
            axis.plot(self.profiles['beamlet grid'], self.profiles[str(comp)]
                      ['density']['m-3'], label=str(comp))
        if hasattr(self, 'x_limits'):
            axis.set_xlim(self.x_limits)
        axis.set_ylabel('Density [1/m3]')
        axis.yaxis.label.set_color('b')
        axis.legend(loc='upper left')
        axis.grid()
        return axis

    def __setup_temperature_axis(self, axis):
        axis.plot(self.profiles['beamlet grid'], self.profiles['electron']['temperature']['eV'], '--', color='r',
                  label='Electron_temperature')
        axis.plot(self.profiles['beamlet grid'], self.profiles['ion1']['temperature']['eV'], '--',
                  label='Ion_temperature', color='m')
        axis.set_ylabel('Temperature [eV]')
        axis.yaxis.label.set_color('r')
        axis.legend(loc='lower right')
        axis.grid()
        return axis

    def __setup_population_axis(self, axis, kind='absolute'):
        pandas_key, axis_name = self.set_axis_parameters(kind)
        for level in range(self.atomic_db.atomic_ceiling):
            label = pandas_key + self.atomic_db.inv_atomic_dict[level]
            axis.plot(self.profiles['beamlet grid'], self.profiles[label], label=label)
        if hasattr(self, 'x_limits'):
            axis.set_xlim(self.x_limits)
        axis.set_yscale('log', nonpositive='clip')
        axis.set_xlabel('Distance [m]')
        axis.set_ylabel(axis_name)
        axis.legend(loc='best', ncol=1)
        self.title = 'Beamlet profiles'
        axis.set_title(self.title)
        axis.grid()
        return axis

    @staticmethod
    def set_axis_parameters(kind):
        assert isinstance(kind, str)
        if kind == 'absolute':
            return 'level ', 'Linear density [1/m]'
        elif kind == 'relative':
            return 'rel.pop ', 'Relative linear density [-]'
        else:
            raise ValueError('Requested plotting format not accepted')

    def setup_benchmark_axis(self, benchmark_profiles, axis):
        benchmark_profiles = benchmark_profiles
        for level in range(self.atomic_db.atomic_ceiling):
            label = 'level ' + str(level)
            axis.plot(benchmark_profiles['beamlet grid'], benchmark_profiles[label], '--', label=label+' ref.')
        return axis

    def save_figure(self, file_path='data/output/beamlet/test_plot.pdf'):
        with PdfPages(file_path) as pdf:
            pdf.savefig()
            d = pdf.infodict()
            d['Title'] = self.title
            d['Keywords'] = 'Source hdf5 file: ' + self.access_path + ', source xml file: ' + self.param_path
            d['ModDate'] = datetime.datetime.today()
