from imas_utility.idsinstance import ImasObject
from utility.exceptions import IdsInstanceLoadError
import numpy as np


class SpectrometerVisibleIds(ImasObject):
    def __init__(self, shot, run, user=None, machine=None):
        super().__init__(shot, run, user, machine)
        self.load_spectrometer_vis_ids()

    def load_spectrometer_vis_ids(self):
        try:
            self.spectro_vis = self.imas_pointer.get('spectrometer_visible')
        except Exception as exc:
            raise IdsInstanceLoadError('No spectrometer_visible IDS was found for shot: '
                                       + str(self.shot) + ' at run ' + str(self.run)) from exc
