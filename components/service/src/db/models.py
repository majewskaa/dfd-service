import uuid
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    email: str = Field(unique=True, index=True)
    password_hash: str
    notify_email: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Job(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: Optional[uuid.UUID] = Field(default=None, foreign_key="user.id", index=True)

    # pending → running → done | failed
    status: str = Field(default="pending", index=True)

    filename: Optional[str] = None
    # Temporary path of the uploaded video, cleaned up after analysis completes.
    upload_path: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # JSON-encoded List[AnalysisSegment] on success.
    result_json: Optional[str] = None
    error: Optional[str] = None

    # One-time token given to anonymous users so they can link the job to a
    # new account at registration time.
    claim_token: str = Field(default_factory=lambda: str(uuid.uuid4()), index=True)
    algorithm: str = Field(default="xception", index=True)
