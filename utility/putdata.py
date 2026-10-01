from utility.accessdata import AccessData
import os


class PutData(AccessData):
    def __init__(self):
        AccessData.__init__(self, data_path_name=None)

    def to_server(self, local_path, server_path=None, server_type='public'):
        if server_path is None:
            if isinstance(local_path, str):
                self.data_path_name = local_path
                self._path_setup()
            else:
                raise TypeError('<local_path> is expected to be of type str. '
                                'The provided data type is: '+str(type(local_path)))
        elif isinstance(server_path, str):
            self._server_path_setup(server_path=server_path)
            self._local_path_setup(local_path=local_path)
        else:
            raise TypeError('<server_path> is expected to be of type str.'
                            'The provided data type is: '+str(type(server_path)))
        if self.check_user_local_data_path():
            if not isinstance(server_type, str):
                raise TypeError('The requested server type is a str information. Provided data is: '
                                + str(type(server_type)))
            if server_type == 'public':
                if self.check_public_server_data_path():
                    print('The file: ' + self.server_public_write_access_path + ' is already on the public server.')
                else:
                    self.connect(protocol='sftp')
                    try:
                        self.sftp.put(self.user_local_data_path, self.server_public_write_access_path)
                        print('Successfully placed: ' + self.user_local_data_path + ' to public server location: '
                              + self.server_public_write_access_path)
                    except FileNotFoundError:
                        self._ensure_dir_server(self.server_public_write_access_path)
                        print('Created folders to place the data in.')
                        self.sftp.put(self.user_local_data_path, self.server_public_write_access_path)
                        print('Successfully placed: ' + self.user_local_data_path + ' to public server location: '
                              + self.server_public_write_access_path)
                    except Exception as exc:
                        print('Could not put file: ' + self.user_local_data_path + ' to public server location: '
                              + self.server_public_write_access_path + ' (' + repr(exc) + ')')
                        raise
                    finally:
                        self.disconnect()
            elif server_type == 'private':
                if self.check_private_server_data_path():
                    print('The file: ' + self.server_private_path + ' is already on the private server.')
                else:
                    self.connect(protocol='sftp')
                    try:
                        self.sftp.put(self.user_local_data_path, self.server_private_path)
                        print('Successfully placed: ' + self.user_local_data_path + ' to private server location: '
                              + self.server_private_path)
                    except FileNotFoundError:
                        self._ensure_dir_server(self.server_private_path)
                        print('Created folders to place the data in.')
                        self.sftp.put(self.user_local_data_path, self.server_private_path)
                        print('Successfully placed: ' + self.user_local_data_path + ' to public server location: '
                              + self.server_private_path)
                    except Exception as exc:
                        print('Could not put file: ' + self.user_local_data_path + ' to private server location: '
                              + self.server_private_path + ' (' + repr(exc) + ')')
                        raise
                    finally:
                        self.disconnect()
            else:
                raise ValueError('The requested server type <'+server_type+'> '
                                                                           'where data is to be ported does not exist!')
        else:
            raise FileNotFoundError('There is no local data at: ' + self.user_local_data_path)

    def delete_from_server(self, data_path, server_type='public'):
        if isinstance(data_path, str):
            self.data_path_name = data_path
            self._server_path_setup(server_path=data_path)
        else:
            raise TypeError('<data_path> is expected to be of type str. '
                            'The provided data type is: '+str(type(data_path)))
        if not isinstance(server_type, str):
            raise TypeError('The requested server type is a str information. Provided data is: '+str(type(server_type)))

        if server_type == 'public':
            if self.check_public_server_data_path():
                self.connect(protocol='sftp')
                self.sftp.remove(self.server_public_write_access_path)
                self.disconnect()
                print('Successfully removed: ' + self.server_public_path + ' from public server.')
            else:
                print('File to be deleted could not be located on public server: ' + self.server_public_path)
        elif server_type == 'private':
            if self.check_private_server_data_path():
                self.connect(protocol='sftp')
                self.sftp.remove(self.server_private_path)
                self.disconnect()
                print('Successfully removed: ' + self.server_private_path + ' from public server.')
            else:
                print('File to be deleted could not be located on private server: ' + self.server_private_path)
        else:
            raise ValueError('The requested server type <'+server_type+'> from which data is to be deleted '
                                                                       'does not exist!')

    def move_data(self, old_path, new_path, data_migration='private-to-private'):
        assert isinstance(old_path, str) and isinstance(new_path, str), 'The input data for the old and new paths ' \
                                                                        'for the data not of <str> format.'
        source_path, target_path = self._set_path_for_data_move(old_path, new_path, data_migration)
        self.connect(protocol='sftp')
        try:
            self.sftp.posix_rename(source_path, target_path)
            print('Successfully moved: ' + source_path + ' to: ' + target_path)
        except FileNotFoundError:
            self._ensure_dir_server(target_path)
            self.sftp.posix_rename(source_path, target_path)
            print('Successfully moved: ' + source_path + ' to: ' + target_path)
        finally:
            self.disconnect()

    def _set_path_for_data_move(self, old_path, new_path, transition):
        assert isinstance(transition, str), 'The expected input type for transition is <str>.'
        if transition == 'private-to-private':
            source = self._set_private_server_path(old_path)
            if not self.check_private_server_data_path(path=source):
                raise FileNotFoundError('The file to be copied: ' + source + ' does not exits!')
            target = self._set_private_server_path(new_path)
            if self.check_private_server_data_path(path=target):
                raise FileExistsError('The file location: ' + target + ' is occupied. New target path required!')
            return source, target
        elif transition == 'private-to-public':
            source = self._set_private_server_path(old_path)
            if not self.check_private_server_data_path(path=source):
                raise FileNotFoundError('The file to be copied: ' + source + ' does not exits!')
            target = self._set_public_server_write_access_path(new_path)
            if self.check_private_server_data_path(path=target):
                raise FileExistsError('The file location: ' + target + ' is occupied. New target path required!')
            return source, target
        elif transition == 'public-to-private':
            source = self._set_public_server_write_access_path(old_path)
            if not self.check_private_server_data_path(path=source):
                raise FileNotFoundError('The file to be copied: ' + source + ' does not exits!')
            target = self._set_private_server_path(new_path)
            if self.check_private_server_data_path(path=target):
                raise FileExistsError('The file location: ' + target + ' is occupied. New target path required!')
            return source, target
        elif transition == 'public-to-public':
            source = self._set_public_server_write_access_path(old_path)
            if not self.check_private_server_data_path(path=source):
                raise FileNotFoundError('The file to be copied: ' + source + ' does not exits!')
            target = self._set_public_server_write_access_path(new_path)
            if self.check_private_server_data_path(path=target):
                raise FileExistsError('The file location: ' + target + ' is occupied. New target path required!')
            return source, target
        else:
            raise ValueError('The transition type: ' + transition + ' is not supported. The following server '
                             'transition types are: <private-to-private>, <private-to-public>, <public-to-private> '
                                                                    'and <public-to-public>')

    def _ensure_dir_server(self, path):
        self.sftp.mkdir(os.path.dirname(path))
