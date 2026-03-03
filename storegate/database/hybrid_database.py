"""HybridDatabase module."""
from storegate.database.database import Database
from storegate.database.zarr_database import ZarrDatabase
from storegate.database.numpy_database import NumpyDatabase


class HybridDatabase(Database):
    """Hybrid database that combines zarr (storage) and numpy (memory) backends."""
    def __init__(self, output_dir, mode='r', chunk=1000):

        self._output_dir = output_dir
        self._chunk = chunk
        self._mode = mode

        self._db = dict()
        self._db['zarr'] = ZarrDatabase(output_dir=output_dir,
                                        chunk=chunk,
                                        mode=mode)
        self._db['numpy'] = NumpyDatabase()
        self._backend = 'zarr'


    def initialize(self, data_id):
        self._db['zarr'].initialize(data_id)
        self._db['numpy'].initialize(data_id)


    def set_backend(self, backend):
        self._backend = backend

    def get_backend(self):
        return self._backend

    def add_data(self, data_id, var_name, data, phase):
        self._db[self._backend].add_data(data_id, var_name, data, phase)


    def update_data(self, data_id, var_name, data, phase, index):
        self._db[self._backend].update_data(data_id, var_name, data, phase, index)


    def get_data(self, data_id, var_name, phase, index):
        return self._db[self._backend].get_data(data_id, var_name, phase, index)


    def delete_data(self, data_id, var_name, phase):
        self._db[self._backend].delete_data(data_id, var_name, phase)


    def get_metadata(self, data_id, phase):
        return self._db[self._backend].get_metadata(data_id, phase)

