import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt

_SECRET_KEY: str = "change-me-in-production"
_ALGORITHM = "HS256"
_EXPIRE_MINUTES: int = 1440  # 24 h


def configure(secret_key: str, expire_minutes: int) -> None:
    global _SECRET_KEY, _EXPIRE_MINUTES
    _SECRET_KEY = secret_key
    _EXPIRE_MINUTES = expire_minutes


def create_access_token(user_id: uuid.UUID) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)


def decode_access_token(token: str) -> Optional[uuid.UUID]:
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
        return uuid.UUID(payload["sub"])
    except (JWTError, KeyError, ValueError):
        return None
