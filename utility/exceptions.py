class IdsError(Exception):
    """Base class for all IDS-related errors."""
    pass


class IdsLoadError(IdsError):
    pass


class IdsAttributeLoadError(IdsError):
    pass


class IdsInstanceLoadError(IdsError):
    pass


class RenateError(Exception):
    """Base class for all RENATE-specific errors."""
    pass


class RenateAuthorizedUserError(RenateError):
    pass


class RenateNotValidTransitionError(RenateError):
    pass


class InputError(Exception):
    pass
