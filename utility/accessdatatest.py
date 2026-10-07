import os
import unittest
from scp import SCPClient
from paramiko.client import SSHClient
from paramiko.rsakey import RSAKey
from paramiko.sftp_client import SFTPClient
from utility.accessdata import AccessData
from utility.accessdata import DEFAULT_SETUP


class AccessDataTest(unittest.TestCase):

    TEST_PATH = 'test_dataset/access_tests/'
    PUBLIC_TEST = 'public_test.txt'
    PRIVATE_TEST = 'private_test.txt'
    PRIVATE_SERVES_ACCESS = 'private_html/renate-od'
    PUBLIC_SERVES_ACCESS = 'public_html/renate-od'
    PUBLIC_SERVER_ADDRESS = 'http://deep.reak.bme.hu/~data/renate-od'
    SERVER_ADDRESS = 'deep.reak.bme.hu'
    SERVER_USER = 'data'
    DUMMY_FOLDER = 'dummy'
    CONTACT_INFO = 'pokol@reak.bme.hu'
    PRIVATE_KEY = 'data/deep_data_key'

    def setUp(self):
        self.access = AccessData(None)

    def tearDown(self):
        del self.access

    def test_public_server_write_access(self):
        self.assertEqual(self.access.server_public_write_access, self.PUBLIC_SERVES_ACCESS,
                         msg='Server public data access does not match expected server public access.')

    def test_private_server_address(self):
        self.assertEqual(self.access.server_private_access, self.PRIVATE_SERVES_ACCESS,
                         msg='Server private data access does not match expected server private access.')

    def test_public_server_address(self):
        self.assertEqual(self.access.server_public_address, self.PUBLIC_SERVER_ADDRESS,
                         msg='Server public data address does not match expected server public address.')

    def test_dummy_folder_name(self):
        self.assertEqual(self.access.dummy_directory, self.DUMMY_FOLDER,
                         msg='Dummy folder name does not match expected dummy folder name.')

    def test_contact_information(self):
        self.assertEqual(self.access.contact_address, self.CONTACT_INFO,
                         msg='Contact information does not match expected contact information.')

    def test_user_local_directory(self):
        self.assertEqual(os.path.normpath(self.access.user_local_data_directory),
                         os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data')),
                         msg='User local data path does not match expected user local data path.')

    def test_user_local_common_directory(self):
        self.assertEqual(os.path.normpath(self.access.common_local_data_directory),
                         os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'common_data')),
                         msg='User local common data path does not match expected user local common data path.')

    def test_private_key(self):
        self.assertIsInstance(self.access.private_key_path, str, msg='The default private key path does not '
                                                                     'match expected private key path.')
        key_path = os.path.join(os.path.dirname(__file__), '..', self.access.private_key_path)
        if os.path.isfile(key_path):
            self.assertIsInstance(self.access.private_key, RSAKey, msg='Default private key does not match expected '
                                  'default private key type.')

    def test_client_setup(self):
        self.assertEqual(self.access.server_address, self.SERVER_ADDRESS, msg='Actual server address does not match '
                                                                              'expected server address.')
        self.assertEqual(self.access.server_user, self.SERVER_USER, msg='Actual server user does not match expected '
                                                                        'server user.')
        if self.access.private_key is not None:
            self.assertIsInstance(self.access.client, SSHClient, msg='Actual connection type does not match '
                                                                     'expected connection type.')
        else:
            self.assertIs(self.access.client, None, msg='If no correct host key is provided client is '
                                                        'expected to be <None>')

    def test_server_connection(self):
        if self.access.client is not None:
            self.access.connect()
            self.assertTrue(self.access.connection, msg='The SSH connection is expected to be active.')
            self.access.disconnect()

    def test_sftp_protocol(self):
        if self.access.client is not None:
            self.access.connect(protocol='sftp')
            self.assertTrue(self.access.communication, msg='The communication protocol is expected to be active.')
            self.assertEqual(self.access.protocol, 'sftp', msg='Expected protocol for server communication is <sftp>')
            self.assertIsInstance(self.access.sftp, SFTPClient, msg='Object to instantiate <sftp> protocol is '
                                                                    'expected to be of SFTPClient')
            self.access.disconnect()

    def test_scp_protocol(self):
        if self.access.client is not None:
            self.access.connect(protocol='scp')
            self.assertTrue(self.access.communication, msg='The communication protocol is expected to be active.')
            self.assertEqual(self.access.protocol, 'scp', msg='Expected protocol for server communication is <scp>')
            self.assertIsInstance(self.access.scp, SCPClient, msg='Object to instantiate <scp> protocol is '
                                                                  'expected to be of SCPClient')
            self.access.disconnect()

    def test_server_path_setup(self):
        path = self.TEST_PATH + self.PRIVATE_TEST
        self.access._server_path_setup(server_path=path)
        self.assertEqual(self.access.server_private_path, self.access.server_private_access + '/' + path,
                         msg='Server private path does not match expected server private path.')
        self.assertEqual(self.access.server_public_path, self.access.server_public_address + '/' + path,
                         msg='Server public path does not match expected server public path.')

    def test_local_path_setup(self):
        path = self.TEST_PATH + self.PRIVATE_TEST
        self.access._local_path_setup(local_path=path)
        self.assertEqual(os.path.normpath(self.access.common_local_data_path),
                         os.path.normpath(os.path.join(self.access.common_local_data_directory, path)),
                         msg='Actual common local path does not match expected common local path.')
        self.assertEqual(os.path.normpath(self.access.user_local_data_path),
                         os.path.normpath(os.path.join(self.access.user_local_data_directory, path)),
                         msg='Actual user local path does not match expected user local path.')
        self.assertEqual(os.path.normpath(self.access.user_local_dummy_path),
                         os.path.normpath(os.path.join(self.access.user_local_data_directory,
                                                       self.access.dummy_directory, path)),
                         msg='Actual user local dummy path does not match expected user local dummy path.')

    def test_public_server_data_check(self):
        self.access._server_path_setup(self.TEST_PATH + self.PUBLIC_TEST)
        self.assertTrue(self.access.check_public_server_data_path(), msg='The datafile: ' + self.PUBLIC_TEST +
                        ' is supposed to be located on the public server at: ' + self.TEST_PATH)
        self.access._server_path_setup(self.TEST_PATH + self.PRIVATE_TEST)
        self.assertFalse(self.access.check_public_server_data_path(), msg='The datafile: ' + self.PRIVATE_TEST +
                         ' is not supposed to be located on the public server at: ' + self.TEST_PATH)

    def test_private_server_data_check(self):
        if self.access.private_key is not None:
            self.access._server_path_setup(self.TEST_PATH + self.PUBLIC_TEST)
            self.assertFalse(self.access.check_private_server_data_path(), msg='The datafile: ' + self.PUBLIC_TEST +
                             'is not supposed to be located on the private server.')
            self.access._server_path_setup(self.TEST_PATH + self.PRIVATE_TEST)
            self.assertTrue(self.access.check_private_server_data_path(), msg='The datafile: ' + self.PUBLIC_TEST +
                            'is supposed to be located on the private server.')

    def test_common_local_data_check(self):
        self.access.common_local_data_directory = os.path.dirname(__file__)
        self.access._local_path_setup(local_path=DEFAULT_SETUP)
        self.assertTrue(self.access.check_common_local_data_path(), msg='DEFAULT SETUP file is expected to exist '
                                                                        'in the location specified.')
        self.assertEqual(self.access.access_path, os.path.join(os.path.dirname(__file__), DEFAULT_SETUP),
                         msg='The actual data path to be accessed does not match to the expected access data path.')
        self.access._local_path_setup(self.PRIVATE_KEY)
        self.assertFalse(self.access.check_common_local_data_path(),
                         msg='PRIVATE_TEST file is expected not to be located in the specified data access path.')

    def test_user_local_data_check(self):
        self.access.user_local_data_directory = os.path.dirname(__file__)
        self.access._local_path_setup(local_path=DEFAULT_SETUP)
        self.assertTrue(self.access.check_user_local_data_path(), msg='DEFAULT SETUP file is expected to exist '
                                                                      'in the location specified.')
        self.assertEqual(self.access.access_path, os.path.join(os.path.dirname(__file__), DEFAULT_SETUP),
                         msg='The actual data path to be accessed does not match to the expected access data path.')
        self.access._local_path_setup(self.PRIVATE_KEY)
        self.assertFalse(self.access.check_user_local_data_path(),
                         msg='PRIVATE_TEST file is expected not to be located in the specified data access path.')

    def test_user_local_dummy_data_check(self):
        self.access.user_local_data_directory = os.path.dirname(__file__)
        self.access.dummy_directory = ''
        self.access._local_path_setup(local_path=DEFAULT_SETUP)
        self.assertTrue(self.access.check_user_local_dummy_path(), msg='DEFAULT SETUP file is expected to exist '
                                                                       'in the location specified.')
        self.access._local_path_setup(self.PRIVATE_KEY)
        self.assertFalse(self.access.check_user_local_dummy_path(),
                         msg='PRIVATE_TEST file is expected not to be located in the specified data access path.')
