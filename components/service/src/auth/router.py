"""
Auth helpers re-exported for backward compatibility.
All endpoint definitions live in service.src.router.router.
"""
from service.src.router.router import get_current_user_id, require_current_user_id

__all__ = ["get_current_user_id", "require_current_user_id"]
