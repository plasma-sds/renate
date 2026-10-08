import os
import urllib.request
import pandas
import h5py
import numpy
from lxml import etree
from utility.accessdata import AccessData


DEFAULT_SETUP = 'getdata_setup.xml'


class GetData(AccessData):
    """
    This class is to access and load data from files. It looks for data in the following order:
    1. Common local data path
    2. User's local data path
    3. Remote private data path (downloads if accessable)
    4. Local user dummy data path
    5. Remote public dummy data path (downloads if accessable)

    data_path_name should include relative path from RENATE-OD root directory
    Paths are read from utility/getdata_setup.xml file.
    """

    def __init__(self,
                 data_path_name=None,
                 data_key=[],
                 data_format="pandas"):
        """
        Init does everything: searches for the requested data, and loads it into the data property.
        :param data_path_name: File name with relative path inside the data directory
        :param data_key: List of keys specifying the groups at the subsequent levels of the hierarchy
        :param data_format: Specifies output data format if ambiguous for the given file type
        """
        AccessData.__init__(self, data_path_name)
        self.data_format = data_format
        if self.data_path_name is None:
            raise ValueError('Variable: data_path_name is not defined!')
        self.data_key = data_key
        self.data = ''
        self.read_data()

    def read_data(self):
        """
        Reads data into the self.data property. Data can be narrowed down by the self.data_key variable.
        :return: True if successful
        """

        if self.get_data():
            if self.data_path_name.endswith('.h5'):
                if self.data_format == "pandas":
                    self.read_h5_to_pandas()
                else:
                    self.read_h5_to_array()
            elif self.data_path_name.endswith('.txt'):
                if self.data_format == "array":
                    self.read_txt_to_array()
                else:
                    self.read_txt_to_str()
            elif self.data_path_name.endswith('.xml'):
                self.read_xml()
            else:
                print('NO data read from file: ' + self.access_path)
        else:
            raise FileNotFoundError('The requested file: ' + self.data_path_name + ' does not exist! Check input!')

    def read_h5_to_pandas(self):
        try:
            if not self.data_key:
                self.data = pandas.read_hdf(self.access_path)
                print('Data read to Pandas DataFrame from HD5 file: ' + self.access_path)
            elif len(self.data_key) > 1:
                raise ValueError('Data could NOT be read to Pandas DataFrame from HD5 file: ' +
                                 self.access_path + " with key: " + str(self.data_key) +
                                 '. Must have only one key maximum!')
            else:
                self.data = pandas.read_hdf(self.access_path, key=self.data_key[0])
                print('Data read to Pandas DataFrame from HD5 file: ' +
                      self.access_path + " with key: " + str(self.data_key[0]))
        except ValueError as exc:
            raise ValueError('Data could NOT be read to Pandas DataFrame from HD5 file: ' +
                             self.access_path + " with key: " + str(self.data_key)) from exc

    def read_h5_to_array(self):
        if not self.data_key:
            raise ValueError('Data could NOT be read to array from HD5 file: ' +
                             self.access_path + '. Key is missing!')
        try:
            with h5py.File(self.access_path, 'r') as file:
                self.data = numpy.array(file[self.data_key[0]])
            print("Data read to array from HD5 file: " + self.access_path + " with key: " + str(self.data_key))
        except ValueError:
            print("Data could NOT be read to array from HD5 file: " + self.access_path +
                  " with key: " + str(self.data_key))
        except AttributeError:
            print("Data could NOT be read to array from HD5 file: " + self.access_path +
                  " with key: " + str(self.data_key) + '. Check if the key sequence fits the groups of the HDF5 file!')

    def read_txt_to_str(self):
        with open(self.access_path, 'r') as file:
            self.data = file.read()
            print('Data read to string from: ' + self.access_path)

    def read_txt_to_array(self):
        self.data = numpy.loadtxt(self.access_path)
        print('Data read to numpy array from : ' + self.access_path)

    def read_xml(self):
        if not self.data_key:
            self.data = etree.parse(self.access_path)
            assert isinstance(self.data, etree._ElementTree)
            print('ElementTree read from: ' + self.access_path)
        else:
            tree = etree.parse(self.access_path)
            element = tree.getroot()
            for name in self.data_key:
                element = element.find(name)
            self.data = element
            assert isinstance(self.data, etree._Element)
            print('Element read from: ' + self.access_path + " with key: " + str(self.data_key))

    def get_data(self):
        """
        Looks for data file at different locations.
        :return: True if successful
        """
        if self.external_path:
            return True
        if self.check_common_local_data_path():
            return True
        elif self.check_user_local_data_path():
            return True
        elif self.check_private_server_data_path():
            self.download_private_data()
            return True
        elif self.check_user_local_dummy_path():
            return True
        elif self.check_public_server_data_path():
            self.download_public_data()
            return True
        else:
            print('Error: No data source available!')
            self.contact_us()
            return False

    def download_private_data(self):
        self.ensure_dir(self.user_local_data_path)
        print('Attempting to download from server: ' + self.server_private_path)
        self.connect(protocol='scp')
        self.scp.get(self.server_private_path, self.user_local_data_path)
        self.disconnect()
        self.access_path = self.user_local_data_path
        print('Private data was downloaded from private server to: ' + self.access_path)

    def download_public_data(self):
        print('Attempting to download dummy data from public server: ' + self.server_public_path)
        self.ensure_dir(self.user_local_dummy_path)
        urllib.request.urlretrieve(self.server_public_path, self.user_local_dummy_path)
        self.access_path = self.user_local_dummy_path
        print('Warning: Dummy data has been downloaded to the user local directory: ' + self.access_path)

    @staticmethod
    def ensure_dir(file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
