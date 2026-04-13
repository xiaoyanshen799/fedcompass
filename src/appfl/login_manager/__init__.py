from .authenticator import BaseAuthenticator
from .naive import NaiveAuthenticator

try:
    from .globus import GlobusLoginManager, GlobusAuthenticator
except ModuleNotFoundError:
    # Globus auth is optional for non-authenticated gRPC runs.
    GlobusLoginManager = None
    GlobusAuthenticator = None
