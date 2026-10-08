class IdsError(Exception):
    """Base class for all IDS-related errors."""
    pass


class IdsLoadError(IdsError):
    """Raised when a full IDS object cannot be located or loaded for the requested shot/run."""
    pass


class IdsAttributeLoadError(IdsError):
    """Raised when a specific attribute (grid, profile, timebase, ...) is missing from an otherwise valid IDS."""
    pass


class IdsInstanceLoadError(IdsError):
    """Raised when a named IDS instance (equilibrium, nbi, spectrometer_visible, ...) cannot be opened."""
    pass


class RenateError(Exception):
    """Base class for all RENATE-specific errors."""
    pass


class RenateAuthorizedUserError(RenateError):
    """Raised when a code-management operation is attempted by a user without the required privileges."""
    pass


class RenateNotValidTransitionError(RenateError):
    """Raised when a requested atomic transition is physically meaningless (e.g. from_level == to_level)."""
    pass


class InputError(Exception):
    """Raised when a public API receives arguments of the wrong type or outside the accepted value set."""
    pass
