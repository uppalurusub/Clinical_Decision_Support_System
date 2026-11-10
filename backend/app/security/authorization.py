"""
app/security/authorization.py

Authorization Layer for the Clinical Decision Support System.

Responsibilities
----------------
1. Authorize authenticated users.
2. Enforce role-based access control (RBAC).
3. Enforce permission-based access control.
4. Support FastAPI authorization dependencies.
5. Provide reusable authorization helpers.
6. Produce security audit logs using Loguru.
7. Never log PHI, credentials, tokens, or clinical content.

Authentication vs Authorization
--------------------------------
Authentication:
    "Who are you?"

Authorization:
    "What are you allowed to do?"

This module assumes authentication has already produced a UserIdentity.

Example:

    current_user = get_current_user(...)
    authorization_manager.require_permission(
        current_user,
        Permission.CLINICAL_DECISION_READ,
    )

Important
---------
Authorization is not the same as clinical safety.

A user may be authorized to access a clinical decision-support endpoint,
but the workflow must still enforce clinical guardrails, human review,
and safety policies.

```

### How it fits your security architecture

The separation should be:

```text
                    ┌─────────────────────┐
                    │   Authentication    │
                    │ authentication.py   │
                    └──────────┬──────────┘
                               │
                               │ UserIdentity
                               ▼
                    ┌─────────────────────┐
                    │    Authorization    │
                    │ authorization.py    │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
               RBAC           ABAC          Policy
              Role →       Contextual      Rules
           Permissions       Access
                 │             │             │
                 └─────────────┼─────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   FastAPI Router    │
                    └─────────────────────┘
```

For example, your clinical decision endpoint can enforce the permission before the request reaches the service:

```python
@router.post("/analyze")
async def analyze(
    request: ClinicalDecisionRequest,
    current_user: UserIdentity = Depends(
        require_permission(
            Permission.CLINICAL_DECISION_ANALYZE
        )
    ),
    service: ClinicalDecisionService = Depends(
        get_clinical_decision_service
    ),
):
    return await service.analyze(request)
```

This gives you the desired flow:

```text
Request
   │
   ▼
Authentication
   │
   │ Who is the user?
   ▼
Authorization
   │
   │ Is the user allowed?
   ▼
Router
   │
   ▼
Service
   │
   ▼
Implementation
   │
   ▼
LangGraph
   │
   ├── Intake Agent
   ├── Context Agent
   ├── Evidence Agent
   ├── Differential Agent
   ├── Missing Information Agent
   ├── Diagnostic Test Agent
   ├── Safety Agent
   ├── Critic Agent
   └── Summary Agent
```

### Important design point

Your `authorization.py` should **not** decide whether a clinical recommendation is medically safe.

There are three distinct layers:

| Layer                  | Question                                               |
| ---------------------- | ------------------------------------------------------ |
| **Authentication**     | Who is the caller?                                     |
| **Authorization**      | Is the caller allowed to perform this operation?       |
| **Clinical Guardrail** | Is this clinical workflow/output safe to proceed with? |

For example:

```text
Doctor
  │
  ├── Authentication ──► Valid
  │
  ├── Authorization ──► CLINICAL_DECISION_ANALYZE = ALLOW
  │
  └── Clinical Guardrail
          │
          ├── Critical missing information
          ├── Safety flag
          ├── Contradictory evidence
          ├── Hallucination
          └── Human review
                  │
                  ▼
             Clinician Review
```

One important follow-up for your existing `authentication.py`: **`UserIdentity` needs a role (and ideally tenant/organization context) for this RBAC implementation to work cleanly.** If your current model does not have `role`, `authorization.py` will correctly deny authorization rather than guessing the user's privileges.

For a production healthcare system, I would next separate this into **`identity.py → authentication.py → authorization.py → rbac.py → abac.py → policies.py → audit.py`**, where `authorization.py` becomes the central PDP (Policy Decision Point), while RBAC and ABAC provide the actual policy inputs.

"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Callable, Iterable, Optional

from fastapi import Depends, HTTPException, status
from app.config.settings import settings
logger = settings.get_logger()

from pydantic import BaseModel, ConfigDict, Field

from app.security.authentication import (
    AuthenticationManager,
    UserIdentity,
    get_current_user,
)


# ============================================================================
# ENUMS
# ============================================================================


class Role(str, Enum):
    """
    Application roles.

    Keep roles coarse-grained.

    Fine-grained authorization should normally be implemented through
    permissions rather than creating a large number of roles.
    """

    PATIENT = "PATIENT"
    DOCTOR = "DOCTOR"
    NURSE = "NURSE"
    ADMIN = "ADMIN"
    CARE_COORDINATOR = "CARE_COORDINATOR"
    SYSTEM = "SYSTEM"


class Permission(str, Enum):
    """
    Fine-grained application permissions.
    """

    # ------------------------------------------------------------------
    # Clinical Decision Support
    # ------------------------------------------------------------------

    CLINICAL_DECISION_READ = "clinical_decision:read"
    CLINICAL_DECISION_ANALYZE = "clinical_decision:analyze"
    CLINICAL_DECISION_REVIEW = "clinical_decision:review"

    # ------------------------------------------------------------------
    # Patient
    # ------------------------------------------------------------------

    PATIENT_READ = "patient:read"
    PATIENT_WRITE = "patient:write"

    # ------------------------------------------------------------------
    # Clinical Records
    # ------------------------------------------------------------------

    CLINICAL_RECORD_READ = "clinical_record:read"
    CLINICAL_RECORD_WRITE = "clinical_record:write"

    # ------------------------------------------------------------------
    # Medications
    # ------------------------------------------------------------------

    MEDICATION_READ = "medication:read"
    MEDICATION_WRITE = "medication:write"

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    DIAGNOSTIC_RESULT_READ = "diagnostic_result:read"
    DIAGNOSTIC_RESULT_WRITE = "diagnostic_result:write"

    # ------------------------------------------------------------------
    # RAG / Evidence
    # ------------------------------------------------------------------

    CLINICAL_EVIDENCE_READ = "clinical_evidence:read"
    CLINICAL_EVIDENCE_MANAGE = "clinical_evidence:manage"

    # ------------------------------------------------------------------
    # Human Review
    # ------------------------------------------------------------------

    HUMAN_REVIEW_READ = "human_review:read"
    HUMAN_REVIEW_APPROVE = "human_review:approve"
    HUMAN_REVIEW_REJECT = "human_review:reject"

    # ------------------------------------------------------------------
    # Administration
    # ------------------------------------------------------------------

    USER_READ = "user:read"
    USER_WRITE = "user:write"

    ROLE_READ = "role:read"
    ROLE_WRITE = "role:write"

    AUDIT_READ = "audit:read"

    SYSTEM_CONFIG_READ = "system_config:read"
    SYSTEM_CONFIG_WRITE = "system_config:write"


class AuthorizationDecision(str, Enum):
    """
    Result of an authorization decision.
    """

    ALLOW = "ALLOW"
    DENY = "DENY"


class AuthorizationReason(str, Enum):
    """
    Machine-readable reason for an authorization decision.
    """

    AUTHENTICATED = "authenticated"
    ROLE_ALLOWED = "role_allowed"
    PERMISSION_ALLOWED = "permission_allowed"
    MISSING_ROLE = "missing_role"
    MISSING_PERMISSION = "missing_permission"
    INACTIVE_USER = "inactive_user"
    INVALID_USER = "invalid_user"
    POLICY_DENIED = "policy_denied"


# ============================================================================
# MODELS
# ============================================================================


class AuthorizationRequest(BaseModel):
    """
    Authorization request.

    This model intentionally contains no clinical payload.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    required_permission: Optional[Permission] = None
    required_role: Optional[Role] = None


class AuthorizationResult(BaseModel):
    """
    Result returned by the authorization engine.
    """

    model_config = ConfigDict(extra="forbid")

    decision: AuthorizationDecision
    reason: AuthorizationReason

    user_id: Optional[str] = None
    role: Optional[Role] = None
    permission: Optional[Permission] = None

    authorized: bool = False


# ============================================================================
# ROLE -> PERMISSION MATRIX
# ============================================================================


ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    # ------------------------------------------------------------------
    # Patient
    # ------------------------------------------------------------------

    Role.PATIENT: frozenset(
        {
            Permission.PATIENT_READ,
            Permission.CLINICAL_RECORD_READ,
            Permission.MEDICATION_READ,
            Permission.DIAGNOSTIC_RESULT_READ,
            Permission.CLINICAL_DECISION_READ,
        }
    ),

    # ------------------------------------------------------------------
    # Doctor
    # ------------------------------------------------------------------

    Role.DOCTOR: frozenset(
        {
            Permission.PATIENT_READ,
            Permission.PATIENT_WRITE,
            Permission.CLINICAL_RECORD_READ,
            Permission.CLINICAL_RECORD_WRITE,
            Permission.MEDICATION_READ,
            Permission.MEDICATION_WRITE,
            Permission.DIAGNOSTIC_RESULT_READ,
            Permission.DIAGNOSTIC_RESULT_WRITE,
            Permission.CLINICAL_DECISION_READ,
            Permission.CLINICAL_DECISION_ANALYZE,
            Permission.CLINICAL_DECISION_REVIEW,
            Permission.CLINICAL_EVIDENCE_READ,
            Permission.HUMAN_REVIEW_READ,
            Permission.HUMAN_REVIEW_APPROVE,
            Permission.HUMAN_REVIEW_REJECT,
        }
    ),

    # ------------------------------------------------------------------
    # Nurse
    # ------------------------------------------------------------------

    Role.NURSE: frozenset(
        {
            Permission.PATIENT_READ,
            Permission.CLINICAL_RECORD_READ,
            Permission.CLINICAL_RECORD_WRITE,
            Permission.MEDICATION_READ,
            Permission.DIAGNOSTIC_RESULT_READ,
            Permission.CLINICAL_DECISION_READ,
            Permission.CLINICAL_DECISION_ANALYZE,
            Permission.CLINICAL_EVIDENCE_READ,
            Permission.HUMAN_REVIEW_READ,
        }
    ),

    # ------------------------------------------------------------------
    # Care Coordinator
    # ------------------------------------------------------------------

    Role.CARE_COORDINATOR: frozenset(
        {
            Permission.PATIENT_READ,
            Permission.CLINICAL_RECORD_READ,
            Permission.MEDICATION_READ,
            Permission.DIAGNOSTIC_RESULT_READ,
            Permission.CLINICAL_DECISION_READ,
            Permission.CLINICAL_EVIDENCE_READ,
            Permission.HUMAN_REVIEW_READ,
        }
    ),

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    Role.ADMIN: frozenset(
        {
            Permission.PATIENT_READ,
            Permission.PATIENT_WRITE,
            Permission.CLINICAL_RECORD_READ,
            Permission.CLINICAL_RECORD_WRITE,
            Permission.MEDICATION_READ,
            Permission.MEDICATION_WRITE,
            Permission.DIAGNOSTIC_RESULT_READ,
            Permission.DIAGNOSTIC_RESULT_WRITE,
            Permission.CLINICAL_DECISION_READ,
            Permission.CLINICAL_DECISION_ANALYZE,
            Permission.CLINICAL_DECISION_REVIEW,
            Permission.CLINICAL_EVIDENCE_READ,
            Permission.CLINICAL_EVIDENCE_MANAGE,
            Permission.HUMAN_REVIEW_READ,
            Permission.HUMAN_REVIEW_APPROVE,
            Permission.HUMAN_REVIEW_REJECT,
            Permission.USER_READ,
            Permission.USER_WRITE,
            Permission.ROLE_READ,
            Permission.ROLE_WRITE,
            Permission.AUDIT_READ,
            Permission.SYSTEM_CONFIG_READ,
            Permission.SYSTEM_CONFIG_WRITE,
        }
    ),

    # ------------------------------------------------------------------
    # System / Service Account
    # ------------------------------------------------------------------

    Role.SYSTEM: frozenset(
        {
            Permission.CLINICAL_DECISION_READ,
            Permission.CLINICAL_DECISION_ANALYZE,
            Permission.CLINICAL_EVIDENCE_READ,
            Permission.DIAGNOSTIC_RESULT_READ,
            Permission.CLINICAL_RECORD_READ,
        }
    ),
}


# ============================================================================
# AUTHORIZATION MANAGER
# ============================================================================


class AuthorizationManager:
    """
    Central authorization engine.

    Responsibilities
    ----------------
    - Check user activity.
    - Check roles.
    - Check permissions.
    - Produce authorization decisions.
    - Provide FastAPI dependency factories.

    This class does NOT:
        - authenticate users
        - validate passwords
        - decode JWTs
        - execute clinical workflows
        - make clinical decisions
    """

    def __init__(
        self,
        role_permissions: Optional[
            dict[Role, frozenset[Permission]]
        ] = None,
    ) -> None:
        self.role_permissions = (
            role_permissions
            if role_permissions is not None
            else ROLE_PERMISSIONS
        )

        logger.info(
            "AuthorizationManager initialized | roles={}",
            len(self.role_permissions),
        )

    # ==================================================================
    # BASIC ROLE CHECK
    # ==================================================================

    def has_role(
        self,
        user: UserIdentity,
        role: Role,
    ) -> bool:
        """
        Return True when the user has the requested role.
        """

        if not self._is_active_user(user):
            return False

        user_role = self._get_user_role(user)

        return user_role == role

    # ==================================================================
    # PERMISSION CHECK
    # ==================================================================

    def has_permission(
        self,
        user: UserIdentity,
        permission: Permission,
    ) -> bool:
        """
        Return True when the user's role grants the permission.
        """

        if not self._is_active_user(user):
            return False

        user_role = self._get_user_role(user)

        if user_role is None:
            return False

        permissions = self.role_permissions.get(
            user_role,
            frozenset(),
        )

        return permission in permissions

    # ==================================================================
    # AUTHORIZE ROLE
    # ==================================================================

    def authorize_role(
        self,
        user: UserIdentity,
        role: Role,
    ) -> AuthorizationResult:
        """
        Evaluate whether the user has the requested role.
        """

        if not self._is_valid_user(user):
            return self._deny(
                user=user,
                reason=AuthorizationReason.INVALID_USER,
                role=role,
            )

        if not self._is_active_user(user):
            return self._deny(
                user=user,
                reason=AuthorizationReason.INACTIVE_USER,
                role=role,
            )

        if self.has_role(user, role):
            logger.info(
                "Authorization allowed | user_id={} | role={}",
                self._safe_user_id(user),
                role.value,
            )

            return AuthorizationResult(
                decision=AuthorizationDecision.ALLOW,
                reason=AuthorizationReason.ROLE_ALLOWED,
                user_id=user.user_id,
                role=role,
                authorized=True,
            )

        logger.warning(
            "Authorization denied | user_id={} | required_role={}",
            self._safe_user_id(user),
            role.value,
        )

        return self._deny(
            user=user,
            reason=AuthorizationReason.MISSING_ROLE,
            role=role,
        )

    # ==================================================================
    # AUTHORIZE PERMISSION
    # ==================================================================

    def authorize_permission(
        self,
        user: UserIdentity,
        permission: Permission,
    ) -> AuthorizationResult:
        """
        Evaluate whether the user has the requested permission.
        """

        if not self._is_valid_user(user):
            return self._deny(
                user=user,
                reason=AuthorizationReason.INVALID_USER,
                permission=permission,
            )

        if not self._is_active_user(user):
            return self._deny(
                user=user,
                reason=AuthorizationReason.INACTIVE_USER,
                permission=permission,
            )

        user_role = self._get_user_role(user)

        if user_role is None:
            return self._deny(
                user=user,
                reason=AuthorizationReason.MISSING_ROLE,
                permission=permission,
            )

        if self.has_permission(user, permission):
            logger.info(
                "Permission allowed | user_id={} | role={} | permission={}",
                self._safe_user_id(user),
                user_role.value,
                permission.value,
            )

            return AuthorizationResult(
                decision=AuthorizationDecision.ALLOW,
                reason=AuthorizationReason.PERMISSION_ALLOWED,
                user_id=user.user_id,
                role=user_role,
                permission=permission,
                authorized=True,
            )

        logger.warning(
            "Permission denied | user_id={} | role={} | permission={}",
            self._safe_user_id(user),
            user_role.value,
            permission.value,
        )

        return self._deny(
            user=user,
            reason=AuthorizationReason.MISSING_PERMISSION,
            permission=permission,
        )

    # ==================================================================
    # REQUIRE ROLE
    # ==================================================================

    def require_role(
        self,
        user: UserIdentity,
        role: Role,
    ) -> UserIdentity:
        """
        Enforce a role.

        Raises:
            HTTPException(403)
        """

        result = self.authorize_role(user, role)

        if not result.authorized:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role privileges.",
            )

        return user

    # ==================================================================
    # REQUIRE PERMISSION
    # ==================================================================

    def require_permission(
        self,
        user: UserIdentity,
        permission: Permission,
    ) -> UserIdentity:
        """
        Enforce a permission.

        Raises:
            HTTPException(403)
        """

        result = self.authorize_permission(
            user,
            permission,
        )

        if not result.authorized:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions.",
            )

        return user

    # ==================================================================
    # MULTIPLE PERMISSIONS
    # ==================================================================

    def has_any_permission(
        self,
        user: UserIdentity,
        permissions: Iterable[Permission],
    ) -> bool:
        """
        Return True if the user has at least one permission.
        """

        return any(
            self.has_permission(user, permission)
            for permission in permissions
        )

    def has_all_permissions(
        self,
        user: UserIdentity,
        permissions: Iterable[Permission],
    ) -> bool:
        """
        Return True if the user has every requested permission.
        """

        return all(
            self.has_permission(user, permission)
            for permission in permissions
        )

    def require_any_permission(
        self,
        user: UserIdentity,
        permissions: Iterable[Permission],
    ) -> UserIdentity:
        """
        Require at least one permission.
        """

        permissions = list(permissions)

        if not self.has_any_permission(user, permissions):
            logger.warning(
                "Authorization denied | user_id={} | reason=missing_any_permission",
                self._safe_user_id(user),
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions.",
            )

        return user

    def require_all_permissions(
        self,
        user: UserIdentity,
        permissions: Iterable[Permission],
    ) -> UserIdentity:
        """
        Require all permissions.
        """

        permissions = list(permissions)

        if not self.has_all_permissions(user, permissions):
            logger.warning(
                "Authorization denied | user_id={} | reason=missing_all_permissions",
                self._safe_user_id(user),
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions.",
            )

        return user

    # ==================================================================
    # FASTAPI DEPENDENCY FACTORIES
    # ==================================================================

    def require_permission_dependency(
        self,
        permission: Permission,
    ) -> Callable:
        """
        Create a FastAPI dependency enforcing one permission.

        Example
        -------
        @router.post("/analyze")
        async def analyze(
            current_user: UserIdentity = Depends(
                authorization_manager.require_permission_dependency(
                    Permission.CLINICAL_DECISION_ANALYZE
                )
            )
        ):
            ...
        """

        async def dependency(
            current_user: UserIdentity = Depends(get_current_user),
        ) -> UserIdentity:

            return self.require_permission(
                current_user,
                permission,
            )

        return dependency

    def require_role_dependency(
        self,
        role: Role,
    ) -> Callable:
        """
        Create a FastAPI dependency enforcing one role.
        """

        async def dependency(
            current_user: UserIdentity = Depends(get_current_user),
        ) -> UserIdentity:

            return self.require_role(
                current_user,
                role,
            )

        return dependency

    def require_any_permission_dependency(
        self,
        permissions: Iterable[Permission],
    ) -> Callable:
        """
        Create a FastAPI dependency requiring any permission.
        """

        permissions = tuple(permissions)

        async def dependency(
            current_user: UserIdentity = Depends(get_current_user),
        ) -> UserIdentity:

            return self.require_any_permission(
                current_user,
                permissions,
            )

        return dependency

    # ==================================================================
    # USER ROLE
    # ==================================================================

    def _get_user_role(
        self,
        user: UserIdentity,
    ) -> Optional[Role]:
        """
        Resolve application role from UserIdentity.

        This implementation expects UserIdentity.role to contain
        the role value.

        If your authentication module currently does not expose role,
        add role information to the authenticated identity or retrieve
        it from the user/identity provider store.
        """

        raw_role = getattr(user, "role", None)

        if raw_role is None:
            return None

        if isinstance(raw_role, Role):
            return raw_role

        try:
            return Role(str(raw_role).upper())
        except ValueError:
            logger.warning(
                "Unknown user role | user_id={}",
                self._safe_user_id(user),
            )
            return None

    # ==================================================================
    # USER VALIDATION
    # ==================================================================

    @staticmethod
    def _is_valid_user(
        user: Optional[UserIdentity],
    ) -> bool:
        return user is not None and bool(
            getattr(user, "user_id", None)
        )

    @staticmethod
    def _is_active_user(
        user: UserIdentity,
    ) -> bool:
        return bool(
            getattr(user, "is_active", False)
        )

    # ==================================================================
    # DENY
    # ==================================================================

    def _deny(
        self,
        user: Optional[UserIdentity],
        reason: AuthorizationReason,
        role: Optional[Role] = None,
        permission: Optional[Permission] = None,
    ) -> AuthorizationResult:
        """
        Build a denied authorization result.
        """

        logger.warning(
            "Authorization denied | user_id={} | reason={} | role={} | permission={}",
            self._safe_user_id(user),
            reason.value,
            role.value if role else None,
            permission.value if permission else None,
        )

        return AuthorizationResult(
            decision=AuthorizationDecision.DENY,
            reason=reason,
            user_id=(
                getattr(user, "user_id", None)
                if user
                else None
            ),
            role=role,
            permission=permission,
            authorized=False,
        )

    # ==================================================================
    # SAFE USER ID
    # ==================================================================

    @staticmethod
    def _safe_user_id(
        user: Optional[UserIdentity],
    ) -> str:
        """
        Return a logging-safe identifier.

        Do not log:
            - patient identifiers
            - usernames
            - email addresses
            - tokens
            - clinical data

        Hashing prevents the raw identifier from appearing in logs.
        """

        if user is None:
            return "unknown"

        raw_user_id = getattr(
            user,
            "user_id",
            None,
        )

        if not raw_user_id:
            return "unknown"

        import hashlib

        return hashlib.sha256(
            str(raw_user_id).encode("utf-8")
        ).hexdigest()[:12]


# ============================================================================
# GLOBAL AUTHORIZATION MANAGER
# ============================================================================


@lru_cache(maxsize=1)
def get_authorization_manager() -> AuthorizationManager:
    """
    Return the application-wide authorization manager.
    """

    manager = AuthorizationManager()

    logger.debug(
        "Authorization manager created"
    )

    return manager


# ============================================================================
# FASTAPI DEPENDENCY HELPERS
# ============================================================================


def require_permission(
    permission: Permission,
) -> Callable:
    """
    Convenience dependency factory.

    Example
    -------

    @router.post("/analyze")
    async def analyze(
        current_user: UserIdentity = Depends(
            require_permission(
                Permission.CLINICAL_DECISION_ANALYZE
            )
        )
    ):
        ...
    """

    manager = get_authorization_manager()

    return manager.require_permission_dependency(
        permission
    )


def require_role(
    role: Role,
) -> Callable:
    """
    Convenience dependency factory.
    """

    manager = get_authorization_manager()

    return manager.require_role_dependency(
        role
    )


def require_any_permission(
    permissions: Iterable[Permission],
) -> Callable:
    """
    Convenience dependency factory.
    """

    manager = get_authorization_manager()

    return manager.require_any_permission_dependency(
        permissions
    )


# ============================================================================
# EXPLICIT AUTHORIZATION HELPERS
# ============================================================================


def authorize_user(
    user: UserIdentity,
    permission: Permission,
) -> AuthorizationResult:
    """
    Explicit authorization helper.

    Useful outside FastAPI route dependencies.
    """

    return get_authorization_manager().authorize_permission(
        user,
        permission,
    )


def user_has_permission(
    user: UserIdentity,
    permission: Permission,
) -> bool:
    """
    Boolean authorization helper.
    """

    return get_authorization_manager().has_permission(
        user,
        permission,
    )


def user_has_role(
    user: UserIdentity,
    role: Role,
) -> bool:
    """
    Boolean role helper.
    """

    return get_authorization_manager().has_role(
        user,
        role,
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================


__all__ = [
    "Role",
    "Permission",
    "AuthorizationDecision",
    "AuthorizationReason",
    "AuthorizationRequest",
    "AuthorizationResult",
    "ROLE_PERMISSIONS",
    "AuthorizationManager",
    "get_authorization_manager",
    "require_permission",
    "require_role",
    "require_any_permission",
    "authorize_user",
    "user_has_permission",
    "user_has_role",
]