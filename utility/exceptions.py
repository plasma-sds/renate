class IdsLoadError(Exception):
    pass


class IdsAttributeLoadError(Exception):
    pass


class IdsInstanceLoadError(Exception):
    pass


class RenateAuthorizedUserError(Exception):
    pass


class RenateNotValidTransitionError(Exception):
    pass

  
class InputError(Exception):
    def __init__(self, message):
        self.message = message
