import pyual
import os
from utility.exceptions import IdsLoadError


class ImasObject:
    def __init__(self, shot, run, user, machine):
        assert isinstance(shot, int)
        self.shot = shot

        assert isinstance(run, int)
        self.run = run

        if user is None:
            self.user = os.getenv('USER')
            self.local_data_base = False
        else:
            assert isinstance(user, str)
            self.user = user
            self.local_data_base = True

        if machine is None:
            self.machine = 'iter'
        else:
            assert isinstance(machine, str)
            self.machine = machine

        self.build_imas_pointer()

    def build_imas_pointer(self):
        try:
            if self.local_data_base is True:
                self.imas_pointer = pyual.Client(self.shot, self.run, user=self.user, machine=self.machine)
            else:
                self.imas_pointer = pyual.Client(self.shot, self.run)
        except Exception as exc:
            raise IdsLoadError('IDS for shot ' + str(self.shot) + ', @ run ' + str(self.run) +
                               ' could not be found. Check input, local idsdb and common idsdb') from exc
