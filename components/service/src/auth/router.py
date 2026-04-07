import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlmodel import Session, select

from service.src.auth import jwt as jwt_utils
from service.src.db.models import Job, User
from service.src.db.session import get_session

router = APIRouter(prefix="/auth", tags=["auth"])
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer = HTTPBearer(auto_error=False)


# ── schemas ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    notify_email: bool = False
    # Optional: link an anonymous job to this new account.
    claim_token: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    notify_email: bool


# ── helpers ───────────────────────────────────────────────────────────────────

def _hash(password: str) -> str:
    return _pwd_context.hash(password)


def _verify(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[uuid.UUID]:
    """Returns the user UUID from the Bearer token, or None if not authenticated."""
    if credentials is None:
        return None
    return jwt_utils.decode_access_token(credentials.credentials)


def require_current_user_id(
    user_id: Optional[uuid.UUID] = Depends(get_current_user_id),
) -> uuid.UUID:
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return user_id


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, session: Session = Depends(get_session)):
    if session.exec(select(User).where(User.email == body.email)).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=body.email,
        password_hash=_hash(body.password),
        notify_email=body.notify_email,
    )
    session.add(user)
    session.flush()  # get user.id before committing

    if body.claim_token:
        job = session.exec(
            select(Job).where(Job.claim_token == body.claim_token)
        ).first()
        if job and job.user_id is None:
            job.user_id = user.id

    session.commit()
    return TokenResponse(access_token=jwt_utils.create_access_token(user.id))


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == body.email)).first()
    if user is None or not _verify(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    return TokenResponse(access_token=jwt_utils.create_access_token(user.id))


@router.get("/me", response_model=UserResponse)
def me(
    user_id: uuid.UUID = Depends(require_current_user_id),
    session: Session = Depends(get_session),
):
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(id=user.id, email=user.email, notify_email=user.notify_email)
