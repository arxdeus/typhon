# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Global configuration (sklearn/_config.py). Typhon has no heterogeneous
# dicts, so the configuration is a record held in a module-level list.

class _Config:
    assume_finite: bool
    working_memory: int
    print_changed_only: bool
    display: str
    pairwise_dist_chunk_size: int
    enable_cython_pairwise_dist: bool
    array_api_dispatch: bool
    transform_output: str
    enable_metadata_routing: bool
    skip_parameter_validation: bool

    def __init__(self):
        self.assume_finite = False
        self.working_memory = 1024
        self.print_changed_only = True
        self.display = "diagram"
        self.pairwise_dist_chunk_size = 256
        self.enable_cython_pairwise_dist = True
        self.array_api_dispatch = False
        self.transform_output = "default"
        self.enable_metadata_routing = False
        self.skip_parameter_validation = False

    def copy(self):
        c = _Config()
        c.assume_finite = self.assume_finite
        c.working_memory = self.working_memory
        c.print_changed_only = self.print_changed_only
        c.display = self.display
        c.pairwise_dist_chunk_size = self.pairwise_dist_chunk_size
        c.enable_cython_pairwise_dist = self.enable_cython_pairwise_dist
        c.array_api_dispatch = self.array_api_dispatch
        c.transform_output = self.transform_output
        c.enable_metadata_routing = self.enable_metadata_routing
        c.skip_parameter_validation = self.skip_parameter_validation
        return c

_global_config = [_Config()]

def get_config() -> _Config:
    """Current configuration (a copy; attributes mirror sklearn's dict keys)."""
    return _global_config[0].copy()

def set_config(assume_finite: Optional[bool] = None, working_memory: Optional[int] = None,
               print_changed_only: Optional[bool] = None, display: Optional[str] = None,
               pairwise_dist_chunk_size: Optional[int] = None,
               enable_cython_pairwise_dist: Optional[bool] = None,
               array_api_dispatch: Optional[bool] = None,
               transform_output: Optional[str] = None,
               enable_metadata_routing: Optional[bool] = None,
               skip_parameter_validation: Optional[bool] = None):
    c = _global_config[0]
    if assume_finite is not None:
        c.assume_finite = assume_finite
    if working_memory is not None:
        c.working_memory = working_memory
    if print_changed_only is not None:
        c.print_changed_only = print_changed_only
    if display is not None:
        c.display = display
    if pairwise_dist_chunk_size is not None:
        c.pairwise_dist_chunk_size = pairwise_dist_chunk_size
    if enable_cython_pairwise_dist is not None:
        c.enable_cython_pairwise_dist = enable_cython_pairwise_dist
    if array_api_dispatch is not None:
        c.array_api_dispatch = array_api_dispatch
    if transform_output is not None:
        c.transform_output = transform_output
    if enable_metadata_routing is not None:
        c.enable_metadata_routing = enable_metadata_routing
    if skip_parameter_validation is not None:
        c.skip_parameter_validation = skip_parameter_validation

class config_context:
    """Context manager for global configuration; restores on exit."""
    _old: _Config
    _assume_finite: Optional[bool]
    _working_memory: Optional[int]
    _print_changed_only: Optional[bool]
    _display: Optional[str]
    _transform_output: Optional[str]
    _skip_parameter_validation: Optional[bool]

    def __init__(self, *, assume_finite: Optional[bool] = None,
                 working_memory: Optional[int] = None,
                 print_changed_only: Optional[bool] = None,
                 display: Optional[str] = None,
                 transform_output: Optional[str] = None,
                 skip_parameter_validation: Optional[bool] = None):
        self._old = get_config()
        self._assume_finite = assume_finite
        self._working_memory = working_memory
        self._print_changed_only = print_changed_only
        self._display = display
        self._transform_output = transform_output
        self._skip_parameter_validation = skip_parameter_validation

    def __enter__(self):
        self._old = get_config()
        set_config(assume_finite=self._assume_finite, working_memory=self._working_memory,
                   print_changed_only=self._print_changed_only, display=self._display,
                   transform_output=self._transform_output,
                   skip_parameter_validation=self._skip_parameter_validation)

    def __exit__(self):
        _global_config[0] = self._old
