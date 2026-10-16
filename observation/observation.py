import numpy
import warnings


class Obs1D:
    """
    One dimensional synthetic observation of beamlet emission. Detectors are placed along the beam axis and each
    collects the photon emission of a single atomic transition emitted within its range.
    Positions are distances along the beam in [m], measured from the beam starting point, which is the first point of
    the beamlet grid (assumed to be 0). Photon fluxes are given in [ph/s].
    """

    def __init__(self, detector_positions, detector_size=None, from_level=None, to_level=None):
        """
        Sets up the detector geometry and the default atomic transition.
        :param detector_positions: Detector centres along the beam in [m], non-negative and strictly increasing.
        :param detector_size: Detector extent along the beam in [m]. None: detector boundaries are halfway between
            neighbouring positions, edge detectors are padded symmetrically. Single number: all detectors have the
            same size, ranges are position -/+ size/2. Array: one size per detector, ranges are position -/+ size/2.
        :param from_level: Upper level of the default transition, e.g. '3n'. Must be given together with to_level.
            None: the default transition of the beam species is used.
        :param to_level: Lower level of the default transition, e.g. '2n'.
        Sets the read-only attributes detector_positions (N), detector_size (N) and detector_ranges (N, 2) in [m],
        where detector_ranges[:, 0] are the lower and detector_ranges[:, 1] the upper detector boundaries.
        """
        self.detector_positions = self.__validate_detector_positions(detector_positions)
        self.detector_size, self.detector_ranges = self.__compute_detector_ranges(self.detector_positions,
                                                                                  detector_size)
        if (from_level is None) != (to_level is None):
            raise ValueError('Both <from_level> and <to_level> must be given to set a default transition, '
                             'or neither to use the default transition of the beam species.')
        if from_level is not None and not (isinstance(from_level, str) and isinstance(to_level, str)):
            raise TypeError('The expected data type for <from_level> and <to_level> is str.')
        self.from_level = from_level
        self.to_level = to_level
        for array in (self.detector_positions, self.detector_size, self.detector_ranges):
            array.flags.writeable = False

    @staticmethod
    def __validate_detector_positions(detector_positions):
        """Returns a validated copy of the detector positions."""
        positions = numpy.array(detector_positions, dtype=float)
        if positions.ndim != 1 or positions.size == 0:
            raise ValueError('<detector_positions> must be a non-empty 1D array.')
        if not numpy.all(numpy.isfinite(positions)):
            raise ValueError('<detector_positions> must contain finite values only.')
        if numpy.any(positions < 0):
            raise ValueError('<detector_positions> are distances along the beam measured from the beam starting '
                             'point and must be non-negative.')
        if numpy.any(numpy.diff(positions) <= 0):
            raise ValueError('<detector_positions> must be strictly increasing.')
        return positions

    @staticmethod
    def __compute_detector_ranges(positions, detector_size):
        """Returns the detector sizes and ranges for the given size mode."""
        if detector_size is None:
            if positions.size < 2:
                raise ValueError('At least two <detector_positions> are needed to derive detector ranges '
                                 'when <detector_size> is None.')
            midpoints = (positions[1:] + positions[:-1]) / 2.
            lower_edges = numpy.concatenate(([positions[0] - (midpoints[0] - positions[0])], midpoints))
            upper_edges = numpy.concatenate((midpoints, [positions[-1] + (positions[-1] - midpoints[-1])]))
            sizes = upper_edges - lower_edges
        else:
            sizes = numpy.array(detector_size, dtype=float)
            if sizes.ndim == 0:
                sizes = numpy.full(positions.shape, float(sizes))
            elif sizes.shape != positions.shape:
                raise ValueError('<detector_size> must be None, a single number or an array of the same size '
                                 'as <detector_positions>.')
            if not numpy.all(numpy.isfinite(sizes)) or numpy.any(sizes <= 0):
                raise ValueError('<detector_size> must contain finite, positive values only.')
            lower_edges = positions - sizes / 2.
            upper_edges = positions + sizes / 2.
        return sizes, numpy.column_stack((lower_edges, upper_edges))

    def _resolve_transition(self, beamlet, from_level=None, to_level=None):
        """Returns the levels and label of the transition: call input, then class default, then beam species default."""
        if (from_level is None) != (to_level is None):
            raise ValueError('Both <from_level> and <to_level> must be given to override the default transition, '
                             'or neither.')
        if from_level is None:
            if self.from_level is not None:
                from_level, to_level = self.from_level, self.to_level
            else:
                from_level, to_level, _, label = beamlet.atomic_db.set_default_atomic_levels()
                return from_level, to_level, label
        return from_level, to_level, from_level + '-->' + to_level

    @staticmethod
    def _get_linear_emission_density(beamlet, from_level, to_level, label):
        """Returns the beamlet grid and the linear emission density in [ph/(s m)], computed only if not present."""
        try:
            emission = beamlet.profiles[label].to_numpy()
        except KeyError:
            beamlet.compute_linear_emission_density(from_level=from_level, to_level=to_level)
            try:
                emission = beamlet.profiles[label].to_numpy()
            except KeyError:
                raise RuntimeError('Linear emission density for transition ' + label + ' could not be computed. '
                                   'Run the beam evolution solver first.') from None
        grid = beamlet.profiles[('beamlet grid', 'distance', 'm')].to_numpy()
        return grid, emission

    @staticmethod
    def _integrate_over_detectors(grid, emission, detector_ranges):
        """
        Exact integral of the piecewise linear emission density between the detector boundaries, evaluated from the
        cumulative trapezoidal integral and its quadratic form within each grid cell. Assumes a strictly increasing
        grid.
        """
        steps = numpy.diff(grid)
        slopes = numpy.diff(emission) / steps
        cumulative = numpy.concatenate(([0.], numpy.cumsum((emission[1:] + emission[:-1]) / 2. * steps)))
        index = numpy.clip(numpy.searchsorted(grid, detector_ranges, side='right') - 1, 0, grid.size - 2)
        offset = detector_ranges - grid[index]
        integral = cumulative[index] + offset * (emission[index] + slopes[index] * offset / 2.)
        return integral[:, 1] - integral[:, 0]

    def _flag_detectors_outside_grid(self, grid, photon_flux):
        """Sets the flux of detectors extending beyond the beamlet grid to NaN and warns."""
        outside = (self.detector_ranges[:, 0] < grid[0]) | (self.detector_ranges[:, 1] > grid[-1])
        if outside.any():
            photon_flux[outside] = numpy.nan
            warnings.warn('Detectors ' + str(numpy.flatnonzero(outside).tolist()) + ' extend beyond the beamlet grid '
                          '(' + str(grid[0]) + ' m - ' + str(grid[-1]) + ' m); their emission is set to NaN.')
        return photon_flux

    @staticmethod
    def _get_beamlet_source(beamlet):
        """Returns the beamlet_source of the beamlet from its parameters, or None if it is not given."""
        try:
            return beamlet.param.getroot().find('body').find('beamlet_source').text
        except AttributeError:
            return None

    def compute_photon_flux(self, beamlets, from_level=None, to_level=None, store=False):
        """
        Computes the photon flux collected by each detector by integrating the linear emission density of the
        observed transition over the detector ranges.
        :param beamlets: A solved Beamlet object or a list of them.
        :param from_level: Upper level of the transition, overrides the default for this call only. Must be given
            together with to_level.
        :param to_level: Lower level of the transition, overrides the default for this call only.
        :param store: If True, the result is stored in photon_flux, the transition labels in observed_transitions and
            the beamlet sources in observed_beamlets, overwriting any previously stored result.
        :return: detector_positions in [m] and photon flux in [ph/s], shaped (N) for a single beamlet or
            (len(beamlets), N) for a list. Detectors extending beyond the beamlet grid are set to NaN with a warning.
        The linear emission density is taken from beamlet.profiles if already present, otherwise it is computed and
        added to beamlet.profiles. An existing emission profile is not recomputed if the beamlet is solved again.
        """
        single = not isinstance(beamlets, (list, tuple))
        if single:
            beamlets = [beamlets]
        photon_flux = numpy.empty((len(beamlets), self.detector_positions.size))
        transitions = []
        for index, beamlet in enumerate(beamlets):
            beamlet_from, beamlet_to, label = self._resolve_transition(beamlet, from_level, to_level)
            grid, emission = self._get_linear_emission_density(beamlet, beamlet_from, beamlet_to, label)
            photon_flux[index] = self._flag_detectors_outside_grid(
                grid, self._integrate_over_detectors(grid, emission, self.detector_ranges))
            transitions.append(label)
        if single:
            photon_flux = photon_flux[0]
        if store:
            self.photon_flux = photon_flux
            self.observed_transitions = transitions
            self.observed_beamlets = [self._get_beamlet_source(beamlet) for beamlet in beamlets]
        return self.detector_positions, photon_flux

    def show(self, beamlet_index=0, display=True):
        """
        Plots a stored photon flux profile against the distance along the beam. Each detector is shown at its
        position with a horizontal bar spanning its range. Detectors with NaN flux are not shown.
        :param beamlet_index: Index of the stored photon flux profile to plot, if more than one is stored.
        :param display: If True, the figure is displayed with matplotlib.pyplot.show().
        :return: The matplotlib figure and axis.
        """
        if not hasattr(self, 'photon_flux'):
            raise ValueError('No photon flux is stored in the observation. '
                             'Call compute_photon_flux() with <store=True> first.')
        photon_flux = numpy.atleast_2d(self.photon_flux)
        if not -photon_flux.shape[0] <= beamlet_index < photon_flux.shape[0]:
            raise IndexError('<beamlet_index> ' + str(beamlet_index) + ' is out of range, '
                             + str(photon_flux.shape[0]) + ' photon flux profile(s) are stored.')
        import matplotlib.pyplot
        figure, axis = matplotlib.pyplot.subplots()
        axis.errorbar(self.detector_positions, photon_flux[beamlet_index],
                      xerr=(self.detector_positions - self.detector_ranges[:, 0],
                            self.detector_ranges[:, 1] - self.detector_positions), fmt='o', capsize=3)
        axis.set_xlabel('Distance along the beam [m]')
        axis.set_ylabel('Photon flux [ph/s]')
        axis.set_title(self.observed_transitions[beamlet_index] + ', beamlet ' + str(beamlet_index) + ': '
                       + str(self.observed_beamlets[beamlet_index]))
        axis.grid()
        if display:
            matplotlib.pyplot.show()
        return figure, axis
