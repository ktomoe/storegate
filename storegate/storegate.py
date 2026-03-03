"""StoreGate module."""

import functools

from storegate import logger, const
from storegate.database import HybridDatabase


class _PhaseAccessor:
    """Intermediate accessor returned by StoreGate[phase].

    Holds phase context without mutating the parent StoreGate instance.
    """

    def __init__(self, storegate, phase):
        self._storegate = storegate
        self._phase = phase

    def __getitem__(self, item):
        if isinstance(item, str):
            return _VarAccessor(self._storegate, self._phase, item)
        raise NotImplementedError(f'item {item} is not supported')

    def __setitem__(self, item, data):
        if not isinstance(item, str):
            raise ValueError(f'item {item} must be str')
        self._storegate.add_data(item, data, phase=self._phase)

    def __delitem__(self, item):
        if not isinstance(item, str):
            raise ValueError(f'item {item} must be str')
        self._storegate.delete_data(item, phase=self._phase)

    def __iter__(self):
        return iter(self._storegate.get_var_names(self._phase))

    def items(self):
        """Yield (var_name, data) pairs for all variables in this phase."""
        for var_name in self:
            yield var_name, self._storegate.get_data(var_name, self._phase, index=None)

    def __contains__(self, item):
        return item in self._storegate.get_var_names(self._phase)

    def __len__(self):
        data_id = self._storegate._data_id
        backend = self._storegate.get_backend()
        if not self._storegate._metadata[data_id]['compiled'][backend]:
            raise ValueError('len() is supported only after compile')
        return self._storegate._metadata[data_id]['sizes'][backend][self._phase]


class _VarAccessor:
    """Intermediate accessor returned by StoreGate[phase][var_name].

    Holds phase and var_name context without mutating the parent StoreGate instance.
    """

    def __init__(self, storegate, phase, var_name):
        self._storegate = storegate
        self._phase = phase
        self._var_name = var_name

    def _normalize_index(self, item):
        if self._phase == 'all' and item == slice(None, None, None):
            return None
        return item

    def __getitem__(self, item):
        if not isinstance(item, (int, slice)):
            raise NotImplementedError(f'item {item} is not supported')
        return self._storegate.get_data(
            var_name=self._var_name, phase=self._phase, index=self._normalize_index(item)
        )

    def __setitem__(self, item, data):
        if not isinstance(item, (int, slice)):
            raise ValueError(f'item {item} must be int or slice')
        self._storegate.update_data(
            var_name=self._var_name, data=data, phase=self._phase, index=self._normalize_index(item)
        )


def require_data_id(method):
    """Raise RuntimeError if self._data_id is not set."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._data_id is None:
            raise RuntimeError(
                f'{method.__name__}() requires data_id to be set. Call set_data_id() first.'
            )
        return method(self, *args, **kwargs)
    return wrapper


class StoreGate:
    """Data management class."""

    def __init__(self, output_dir, mode='r', chunk=1000, data_id=None):
        """Initialize the storegate and the zarr architecture."""

        self._db = HybridDatabase(output_dir=output_dir, mode=mode, chunk=chunk)
        self._data_id = None
        self._metadata = {}

        if data_id is not None:
            self.set_data_id(data_id)


    @require_data_id
    def __getitem__(self, item):
        """Return a phase accessor for chained access: sg[phase][var_name][index]."""
        if (item in const.PHASES) or (item == 'all'):
            return _PhaseAccessor(self, item)

        raise NotImplementedError(f'item {item} is not supported')


    ##########################################################################
    # Public user APIs
    ##########################################################################
    @property
    def data_id(self):
        """Return the current data_id."""
        return self._data_id

    def set_data_id(self, data_id):
        """Set the default ``data_id`` and initialize the zarr."""
        self._data_id = data_id
        self._db.initialize(data_id)

        if data_id not in self._metadata:
            self._metadata[self._data_id] = {'compiled': {'zarr': False, 'numpy': False},
                                             'sizes': {'zarr': {}, 'numpy': {}}}


    @require_data_id
    def set_backend(self, backend):
        """Set backend mode of hybrid architecture."""
        if backend not in ['numpy', 'zarr']:
            raise ValueError(f'Unsupported backend: "{backend}". Use "numpy" or "zarr".')

        self._db.set_backend(backend)


    @require_data_id
    def get_backend(self):
        return self._db.get_backend()


    @require_data_id
    def add_data(self, var_name, data, phase):
        self._db.add_data(self._data_id, var_name, data, phase)
        self._metadata[self._data_id]['compiled'][self.get_backend()] = False


    @require_data_id
    def add_data_splits(self, var_name, *, train=None, valid=None, test=None):
        """Add data for multiple phases in a single call.

        Each phase argument is optional; omitted phases are skipped.

        Args:
            var_name (str): Variable name to register.
            train: Data for the train phase.
            valid: Data for the valid phase.
            test:  Data for the test phase.

        Examples:
            >>> sg.add_data_splits('x', train=x_train, valid=x_valid, test=x_test)
        """
        for phase, data in zip(const.PHASES, (train, valid, test)):
            if data is not None:
                self.add_data(var_name, data, phase)


    @require_data_id
    def update_data(self, var_name, data, phase, index=None):
        """Update data in storegate with given options."""
        self._db.update_data(self._data_id, var_name, data, phase, index)
        self._metadata[self._data_id]['compiled'][self.get_backend()] = False


    @require_data_id
    def get_data(self, var_name, phase, index=None):
        """Retrieve data from storegate with given options."""
        return self._db.get_data(self._data_id, var_name, phase, index)


    @require_data_id
    def delete_data(self, var_name, phase):
        """Delete data associated with var_names."""
        if phase == 'all':
            for iphase in const.PHASES:
                self._db.delete_data(self._data_id, var_name, iphase)

        else:
            self._db.delete_data(self._data_id, var_name, phase)
        self._metadata[self._data_id]['compiled'][self.get_backend()] = False


    @require_data_id
    def get_var_names(self, phase):
        """Returns registered var_names for given phase."""
        metadata = self._db.get_metadata(self._data_id, phase)
        return list(metadata.keys())


    @require_data_id
    def copy_to_memory(self, var_name, phase, output_var_name=None):
        """Copy data from storage to memory."""
        if output_var_name is None:
            output_var_name = var_name

        tmp_backend = self.get_backend()
        try:
            self.set_backend('numpy')
            if output_var_name in self.get_var_names(phase):
                raise ValueError(f'{output_var_name} already exists in memory. Delete first or use a different output_var_name.')

            self.set_backend('zarr')
            tmp_data = self.get_data(var_name, phase=phase)
            self.set_backend('numpy')
            self.add_data(output_var_name, tmp_data, phase)
        finally:
            self.set_backend(tmp_backend)


    @require_data_id
    def copy_to_storage(self, var_name, phase, output_var_name=None):
        """Copy data from memory to storage."""
        if output_var_name is None:
            output_var_name = var_name

        tmp_backend = self.get_backend()
        try:
            self.set_backend('zarr')
            if output_var_name in self.get_var_names(phase):
                raise ValueError(f'{output_var_name} already exists in storage. Delete first or use a different output_var_name.')

            self.set_backend('numpy')
            tmp_data = self.get_data(var_name, phase=phase)
            self.set_backend('zarr')
            self.add_data(output_var_name, tmp_data, phase)
        finally:
            self.set_backend(tmp_backend)


    @require_data_id
    def compile(self, show_info=False):
        """Check if registered data are valid.

        Validates that all variables within each phase have the same number of events.
        Note: consistency across phases (e.g. train vs valid) is intentionally not checked,
        as each phase may have a different number of events by design.
        """

        num_events = []
        for phase in const.PHASES:
            metadata = self._db.get_metadata(self._data_id, phase)

            phase_events = []
            for data in metadata.values():
                phase_events.append(data['total_events'])

            if len(set(phase_events)) > 1:
                raise ValueError(f'Number of events are not consistent {metadata}')

            if phase_events:
                num_events.append(phase_events[0])
            else:
                num_events.append(None)

        self._metadata[self._data_id]['compiled'][self.get_backend()] = True

        for phase, events in zip(const.PHASES, num_events):
            self._metadata[self._data_id]['sizes'][self.get_backend()][phase] = events

        if show_info:
            self.show_info()


    @require_data_id
    def show_info(self):
        """Show information currently registered in storegate."""
        is_compiled = self._metadata[self._data_id]['compiled'][self.get_backend()]
        header = f'StoreGate data_id : {self._data_id}, compiled : {is_compiled}'

        names = ['phase', 'backend', 'var_name', 'var_type', 'total_events', 'var_shape']

        table_data = []
        for phase in const.PHASES:
            metadata = self._db.get_metadata(self._data_id, phase)
            if not metadata:
                continue

            for var_name, data in metadata.items():
                backend = data['backend']
                dtype = data['type']
                total_events = str(data["total_events"])
                shape = f'{data["shape"]}'

                table_data.append([phase, backend, var_name, dtype, total_events, shape])
            table_data.append('-')

        logger.table(names, table_data, header=header)
