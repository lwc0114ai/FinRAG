from fastapi import Header

from finrag.common.config import get_settings
from finrag.common.exceptions import AuthError


def optional_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> None:
    s = get_settings()
    if not s.api_key:
        return
    if not x_api_key or x_api_key != s.api_key:
        raise AuthError("Invalid or missing X-API-Key")
