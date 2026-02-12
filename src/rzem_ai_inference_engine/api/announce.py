"""mDNS/DNS-SD network announcement for local service discovery.

Registers the server as a ``_rzem-ai._tcp.local.`` service so that
client applications on the same LAN can find it automatically without
the user having to enter host/port details manually.

Uses the `zeroconf <https://pypi.org/project/zeroconf/>`_ library which
implements multicast DNS (RFC 6762) and DNS-based Service Discovery
(RFC 6763) — the same mechanism behind Apple Bonjour, Chromecast
discovery, etc.
"""

from __future__ import annotations

import socket
from typing import Any

from loguru import logger


SERVICE_TYPE = "_rzem-ai._tcp.local."
SERVICE_NAME = "RZEM AI Inference Engine._rzem-ai._tcp.local."


def _get_local_ip(target_host: str) -> str:
    """Return the best local IP address for announcement.

    When the server binds to ``0.0.0.0`` we need to figure out the
    actual LAN address.  We do this by opening a UDP socket to a
    non-routable target — the OS routing table picks the right
    source address without sending any traffic.

    If the server binds to a specific interface address we just use
    that directly (unless it's ``127.0.0.1``).
    """
    if target_host not in ("0.0.0.0", "::"):
        return target_host

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Doesn't actually send anything — just triggers route lookup
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


class ServiceAnnouncer:
    """Register and unregister the inference server on the local network.

    Parameters
    ----------
    host:
        The address the HTTP server is bound to (e.g. ``"0.0.0.0"``).
    port:
        The port the HTTP server is listening on.
    properties:
        Extra key/value metadata to include in the DNS-SD TXT record.
        Values are encoded as UTF-8 bytes.  Useful for advertising the
        API version, device type, etc.
    """

    def __init__(
        self,
        host: str,
        port: int,
        properties: dict[str, str] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._properties = properties or {}
        self._zeroconf: Any = None
        self._info: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self) -> None:
        """Announce the service on the local network via mDNS."""
        try:
            from zeroconf import ServiceInfo, Zeroconf
        except ImportError:
            logger.warning(
                "zeroconf package not installed — skipping network announcement. "
                "Install it with: pip install zeroconf"
            )
            return

        ip = _get_local_ip(self._host)

        if ip == "127.0.0.1":
            logger.info(
                "Server bound to localhost — skipping network announcement "
                "(use --host 0.0.0.0 to announce on the LAN)"
            )
            return

        self._info = ServiceInfo(
            type_=SERVICE_TYPE,
            name=SERVICE_NAME,
            addresses=[socket.inet_aton(ip)],
            port=self._port,
            properties={k: v.encode() for k, v in self._properties.items()},
            server=f"{socket.gethostname()}.local.",
        )

        self._zeroconf = Zeroconf()
        self._zeroconf.register_service(self._info)
        logger.info(
            f"Service announced on LAN: {SERVICE_TYPE} at {ip}:{self._port}"
        )

    def unregister(self) -> None:
        """Remove the service announcement and clean up."""
        if self._zeroconf is not None and self._info is not None:
            logger.info("Removing network service announcement")
            self._zeroconf.unregister_service(self._info)
            self._zeroconf.close()
            self._zeroconf = None
            self._info = None
