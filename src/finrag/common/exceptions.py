class FinRagError(Exception):
    code = "finrag_error"
    status_code = 400

    def __init__(self, message: str = "error") -> None:
        self.message = message
        super().__init__(message)


class NotFoundError(FinRagError):
    code = "not_found"
    status_code = 404


class AuthError(FinRagError):
    code = "unauthorized"
    status_code = 401


class BadRequestError(FinRagError):
    code = "bad_request"
    status_code = 400
