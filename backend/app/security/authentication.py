"""
app/security/authentication.py

Authentication module for the Clinical Decision Support application.

Responsibilities:
    - Authenticate users
    - Validate credentials
    - Create access tokens
    - Decode and validate access tokens
    - Provide FastAPI authentication dependencies
    - Identify the authenticated user
    - Enforce token expiration

Non-responsibilities:
    - Role authorization
    - Permission checking
    - ABAC policy evaluation
    - Clinical authorization
    - PHI access decisions

Recommended architecture:

    Request
       |
       v
    Authentication
       |
       v
    Identity
       |
       v
    RBAC / ABAC
       |
       v
    Clinical API

Security:
    - Never log passwords
    - Never log access tokens
    - Never log refresh tokens
    - Never log Authorization headers
    - Never log PHI
    - Never log complete JWT payloads
    - Log only safe identifiers and authentication status
"""

from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from jose import JWTError, jwt

from app.config.settings import settings
logger = settings.get_logger()

from passlib.context import CryptContext
from pydantic import BaseModel, Field


# =============================================================================
# Configuration
# =============================================================================

JWT_ALGORITHM = os.getenv(
    "JWT_ALGORITHM",
    "HS256",
)

JWT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
)

ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv(
        "ACCESS_TOKEN_EXPIRE_MINUTES",
        "30",
    )
)

ISSUER = os.getenv(
    "JWT_ISSUER",
    "clinical-decision-support",
)

AUDIENCE = os.getenv(
    "JWT_AUDIENCE",
    "clinical-api",
)


# =============================================================================
# Password Hashing
# =============================================================================

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


# =============================================================================
# HTTP Authentication Scheme
# =============================================================================

security = HTTPBearer(
    auto_error=False,
)


# =============================================================================
# Enumerations
# =============================================================================


class AuthenticationStatus(str, Enum):
    """
    Authentication status.
    """

    AUTHENTICATED = "authenticated"
    INVALID_CREDENTIALS = "invalid_credentials"
    EXPIRED = "expired"
    INVALID_TOKEN = "invalid_token"
    INACTIVE_USER = "inactive_user"


# =============================================================================
# Models
# =============================================================================


class UserIdentity(BaseModel):
    """
    Authenticated user identity.

    This represents identity only.

    Roles and permissions should be resolved by the authorization
    layer.
    """

    user_id: str = Field(
        ...,
        description="Unique user identifier.",
    )

    username: str = Field(
        ...,
        description="Authenticated username.",
    )

    email: Optional[str] = Field(
        default=None,
        description="User email address.",
    )

    is_active: bool = Field(
        default=True,
        description="Whether the user account is active.",
    )


class TokenPayload(BaseModel):
    """
    Minimal JWT payload.

    Avoid putting PHI or unnecessary personal information into JWTs.
    """

    sub: str = Field(
        ...,
        description="User identifier.",
    )

    username: str = Field(
        ...,
        description="Username.",
    )

    iss: str = Field(
        ...,
        description="Token issuer.",
    )

    aud: str = Field(
        ...,
        description="Token audience.",
    )

    exp: int = Field(
        ...,
        description="Expiration timestamp.",
    )

    iat: int = Field(
        ...,
        description="Issued-at timestamp.",
    )

    jti: str = Field(
        ...,
        description="Unique token identifier.",
    )


class LoginRequest(BaseModel):
    """
    Login request.

    Password is accepted from the API but must never be logged.
    """

    username: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    password: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )


class TokenResponse(BaseModel):
    """
    Access-token response.
    """

    access_token: str

    token_type: str = "bearer"

    expires_in: int


# =============================================================================
# Authentication Manager
# =============================================================================


class AuthenticationManager:
    """
    Authentication service.

    Production implementations should replace the example user lookup
    with a secure database/identity-provider integration.
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = JWT_ALGORITHM,
        access_token_expire_minutes: int = (
            ACCESS_TOKEN_EXPIRE_MINUTES
        ),
    ) -> None:

        self.secret_key = (
            secret_key
            or JWT_SECRET_KEY
        )

        self.algorithm = algorithm
        self.access_token_expire_minutes = (
            access_token_expire_minutes
        )

        self._validate_configuration()

        logger.info(
            "AuthenticationManager initialized | algorithm={} | "
            "token_expiration_minutes={}",
            self.algorithm,
            self.access_token_expire_minutes,
        )

    # =========================================================================
    # Configuration
    # =========================================================================

    def _validate_configuration(self) -> None:
        """
        Validate authentication configuration.
        """

        if not self.secret_key:

            raise RuntimeError(
                "JWT_SECRET_KEY is not configured."
            )

        if len(self.secret_key) < 32:

            raise RuntimeError(
                "JWT_SECRET_KEY must contain at least 32 characters."
            )

    # =========================================================================
    # Password Hashing
    # =========================================================================

    @staticmethod
    def hash_password(
        password: str,
    ) -> str:
        """
        Hash a password using bcrypt.

        Password itself is never logged.
        """

        if not password:
            raise ValueError(
                "Password cannot be empty."
            )

        return pwd_context.hash(
            password
        )

    @staticmethod
    def verify_password(
        plain_password: str,
        password_hash: str,
    ) -> bool:
        """
        Verify a password against its hash.
        """

        if not plain_password:
            return False

        if not password_hash:
            return False

        try:

            return pwd_context.verify(
                plain_password,
                password_hash,
            )

        except Exception:

            logger.warning(
                "Password verification failed due to "
                "hash verification error"
            )

            return False

    # =========================================================================
    # Authentication
    # =========================================================================

    async def authenticate(
        self,
        username: str,
        password: str,
    ) -> Optional[UserIdentity]:
        """
        Authenticate a user.

        Replace `_get_user_by_username()` with your database or
        enterprise identity provider.

        Returns:
            UserIdentity when authentication succeeds.
            None otherwise.
        """

        safe_username = self._safe_identifier(
            username
        )

        logger.info(
            "Authentication attempt | username={}",
            safe_username,
        )

        try:

            user_record = (
                await self._get_user_by_username(
                    username
                )
            )

            if not user_record:

                logger.warning(
                    "Authentication failed | "
                    "username={} | reason=unknown_user",
                    safe_username,
                )

                return None

            if not user_record.get(
                "is_active",
                True,
            ):

                logger.warning(
                    "Authentication failed | "
                    "username={} | reason=inactive_user",
                    safe_username,
                )

                return None

            password_hash = user_record.get(
                "password_hash"
            )

            if not password_hash:

                logger.error(
                    "Authentication failed | "
                    "username={} | reason=missing_password_hash",
                    safe_username,
                )

                return None

            password_valid = (
                self.verify_password(
                    password,
                    password_hash,
                )
            )

            if not password_valid:

                logger.warning(
                    "Authentication failed | "
                    "username={} | reason=invalid_password",
                    safe_username,
                )

                return None

            identity = UserIdentity(
                user_id=str(
                    user_record["user_id"]
                ),
                username=str(
                    user_record["username"]
                ),
                email=user_record.get(
                    "email"
                ),
                is_active=bool(
                    user_record.get(
                        "is_active",
                        True,
                    )
                ),
            )

            logger.info(
                "Authentication successful | username={}",
                safe_username,
            )

            return identity

        except Exception:

            logger.exception(
                "Authentication error | username={}",
                safe_username,
            )

            raise

    # =========================================================================
    # Access Token
    # =========================================================================

    def create_access_token(
        self,
        identity: UserIdentity,
    ) -> str:
        """
        Create a signed JWT access token.

        Do not place:
            - PHI
            - clinical information
            - passwords
            - secrets
            - unnecessary personal information

        into the token.
        """

        now = datetime.now(
            timezone.utc
        )

        expiration = (
            now
            + timedelta(
                minutes=self.access_token_expire_minutes
            )
        )

        token_id = secrets.token_urlsafe(
            32
        )

        payload = {
            "sub": identity.user_id,
            "username": identity.username,
            "iss": ISSUER,
            "aud": AUDIENCE,
            "iat": int(
                now.timestamp()
            ),
            "exp": int(
                expiration.timestamp()
            ),
            "jti": token_id,
        }

        token = jwt.encode(
            payload,
            self.secret_key,
            algorithm=self.algorithm,
        )

        logger.info(
            "Access token created | username={} | expires_in_minutes={}",
            self._safe_identifier(
                identity.username
            ),
            self.access_token_expire_minutes,
        )

        return token

    # =========================================================================
    # Token Validation
    # =========================================================================

    def decode_access_token(
        self,
        token: str,
    ) -> Optional[TokenPayload]:
        """
        Decode and validate JWT access token.

        Token value is never logged.
        """

        if not token:

            logger.warning(
                "Token validation failed | reason=empty_token"
            )

            return None

        try:

            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[
                    self.algorithm
                ],
                issuer=ISSUER,
                audience=AUDIENCE,
            )

            token_payload = TokenPayload(
                sub=str(
                    payload["sub"]
                ),
                username=str(
                    payload["username"]
                ),
                iss=str(
                    payload["iss"]
                ),
                aud=str(
                    payload["aud"]
                ),
                exp=int(
                    payload["exp"]
                ),
                iat=int(
                    payload["iat"]
                ),
                jti=str(
                    payload["jti"]
                ),
            )

            logger.debug(
                "Access token validated | username={}",
                self._safe_identifier(
                    token_payload.username
                ),
            )

            return token_payload

        except JWTError as exc:

            # Do not log the token or decoded claims.
            logger.warning(
                "Access token validation failed | "
                "error_type={}",
                type(exc).__name__,
            )

            return None

        except Exception:

            logger.exception(
                "Unexpected access token validation error"
            )

            return None

    # =========================================================================
    # Current User
    # =========================================================================

    async def get_current_user(
        self,
        credentials: Optional[
            HTTPAuthorizationCredentials
        ] = Depends(security),
    ) -> UserIdentity:
        """
        FastAPI dependency that authenticates the current request.
        """

        if credentials is None:

            logger.warning(
                "Authentication rejected | reason=missing_credentials"
            )

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required.",
                headers={
                    "WWW-Authenticate": "Bearer"
                },
            )

        if credentials.scheme.lower() != "bearer":

            logger.warning(
                "Authentication rejected | reason=invalid_scheme"
            )

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme.",
                headers={
                    "WWW-Authenticate": "Bearer"
                },
            )

        token_payload = (
            self.decode_access_token(
                credentials.credentials
            )
        )

        if token_payload is None:

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired access token.",
                headers={
                    "WWW-Authenticate": "Bearer"
                },
            )

        user = await self._get_user_by_id(
            token_payload.sub
        )

        if not user:

            logger.warning(
                "Authentication rejected | reason=user_not_found"
            )

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authenticated user was not found.",
                headers={
                    "WWW-Authenticate": "Bearer"
                },
            )

        if not user.get(
            "is_active",
            True,
        ):

            logger.warning(
                "Authentication rejected | reason=inactive_user"
            )

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is inactive.",
                headers={
                    "WWW-Authenticate": "Bearer"
                },
            )

        identity = UserIdentity(
            user_id=str(
                user["user_id"]
            ),
            username=str(
                user["username"]
            ),
            email=user.get(
                "email"
            ),
            is_active=bool(
                user.get(
                    "is_active",
                    True,
                )
            ),
        )

        logger.debug(
            "Current user authenticated | username={}",
            self._safe_identifier(
                identity.username
            ),
        )

        return identity

    # =========================================================================
    # Database / Identity Provider Integration
    # =========================================================================

    async def _get_user_by_username(
        self,
        username: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user from the identity store.

        Replace this method with:
            - PostgreSQL
            - MySQL
            - Azure AD / Entra ID
            - Okta
            - Auth0
            - Keycloak
            - Enterprise SSO
            - Other approved IdP

        The example implementation intentionally does not contain
        hard-coded production credentials.
        """

        raise NotImplementedError(
            "Implement user lookup using the application's "
            "database or enterprise identity provider."
        )

    async def _get_user_by_id(
        self,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user by ID.

        Replace with the application's identity store.
        """

        raise NotImplementedError(
            "Implement user lookup by user ID."
        )

    # =========================================================================
    # Login
    # =========================================================================

    async def login(
        self,
        request: LoginRequest,
    ) -> TokenResponse:
        """
        Authenticate a user and issue an access token.
        """

        identity = await self.authenticate(
            username=request.username,
            password=request.password,
        )

        if identity is None:

            # Do not reveal whether username or password was incorrect.
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password.",
                headers={
                    "WWW-Authenticate": "Bearer"
                },
            )

        access_token = (
            self.create_access_token(
                identity
            )
        )

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=(
                self.access_token_expire_minutes
                * 60
            ),
        )

    # =========================================================================
    # Utility
    # =========================================================================

    @staticmethod
    def _safe_identifier(
        value: Optional[str],
    ) -> str:
        """
        Return a safe, non-sensitive identifier for logs.

        This avoids writing full usernames/emails into logs.
        """

        if not value:
            return "unknown"

        value = str(value)

        digest = hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()

        return digest[:12]


# =============================================================================
# Global Authentication Manager
# =============================================================================

authentication_manager = AuthenticationManager()


# =============================================================================
# FastAPI Dependency
# =============================================================================


async def get_current_user(
    credentials: Optional[
        HTTPAuthorizationCredentials
    ] = Depends(security),
) -> UserIdentity:
    """
    FastAPI dependency for retrieving the authenticated user.

    Usage:

        @router.get("/protected")
        async def protected(
            current_user: UserIdentity = Depends(
                get_current_user
            )
        ):
            ...
    """

    return await authentication_manager.get_current_user(
        credentials
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def hash_password(
    password: str,
) -> str:
    """
    Hash a password.
    """

    return authentication_manager.hash_password(
        password
    )


def verify_password(
    plain_password: str,
    password_hash: str,
) -> bool:
    """
    Verify a password.
    """

    return authentication_manager.verify_password(
        plain_password,
        password_hash,
    )


def create_access_token(
    identity: UserIdentity,
) -> str:
    """
    Create an access token.
    """

    return authentication_manager.create_access_token(
        identity
    )