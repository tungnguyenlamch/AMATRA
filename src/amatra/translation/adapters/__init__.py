"""Importing this package triggers registration of all bundled adapters.

Add a new ``from . import <module>`` line here when you introduce a new adapter
file. Keep ``research_adapter`` above the concrete adapters so the shared
wrapper class is importable by all of them.
"""
from __future__ import annotations

from . import research_adapter  # noqa: F401
from . import elan_mt  # noqa: F401
from . import google  # noqa: F401
from . import llm  # noqa: F401
