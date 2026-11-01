"""SSH client abstraction for V100 monitoring.

This module contains the :class:`V100SSHClient` which encapsulates the
Paramiko connection logic used by the GUI. Keeping the SSH logic in a
dedicated module simplifies unit testing and makes it possible to add a
CLI or CLI‑only backend without touching the GUI code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import paramiko

__all__ = ["V100SSHClient", "parse_nvidia_smi", "NvidiaSMIError"]

_log = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Container for metrics for a single GPU."""

    gpu_util: float
    vram_used: float
    vram_total: float
    temp: float
    power: float


class NvidiaSMIError(RuntimeError):
    """Raised when parsing or execution of ``nvidia-smi`` fails."""


class V100SSHClient:
    """Simple wrapper around :class:`paramiko.SSHClient`.

    Parameters
    ----------
    host, user, password, port, key_filename
        Connection details.  ``port`` may be ``int`` or ``str``.
    timeout
        Connection timeout in seconds.
    """

    def __init__(self, host: str, user: str, password: Optional[str] = None,
                 port: int | str = 22, key_filename: Optional[str] = None,
                 timeout: float = 10.0):
        self.host = host
        self.user = user
        self.password = password
        self.port = int(port) if isinstance(port, str) else port
        self.key_filename = key_filename
        self.timeout = timeout
        self._client: Optional[paramiko.SSHClient] = None

    # ------------------------------------------------------------------ connection
    def _connect(self) -> None:
        if self._client and self._client.get_transport().is_active():
            return
        self._client = paramiko.SSHClient()
        # In a production tool we would use a known‑hosts file.
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kwargs: Dict[str, object] = {
            "hostname": self.host,
            "username": self.user,
            "port": self.port,
            "timeout": self.timeout,
            "banner_timeout": 15,
            "auth_timeout": 15,
        }
        if self.key_filename:
            connect_kwargs["key_filename"] = self.key_filename
            connect_kwargs["allow_agent"] = False
        if self.password:
            connect_kwargs["password"] = self.password
        self._client.connect(**connect_kwargs)

    def run_nvidia_smi(self) -> str:
        """Execute ``nvidia-smi`` and return its stdout as a string.

        Raises :class:`NvidiaSMIError` on any failure.
        """
        self._connect()
        try:
            stdin, stdout, stderr = self._client.exec_command(
                "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,fb_memory_usage.used,fb_memory_usage.total"  # noqa: E501
                + " --format=csv,noheader,nounits",
                timeout=self.timeout,
            )
            out = stdout.read().decode("utf-8").strip()
            err = stderr.read().decode("utf-8").strip()
            if err:
                raise NvidiaSMIError(f"SSH command failed: {err}")
            if not out:
                raise NvidiaSMIError("nvidia-smi returned no output")
            return out
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise NvidiaSMIError(str(exc)) from exc

    # ------------------------------------------------------------------ parsing
    @staticmethod
    def parse_nvidia_smi(output: str) -> List[GPUInfo]:
        """Parse the CSV output of ``nvidia-smi``.

        The command used in :meth:`run_nvidia_smi` returns a single line per
        GPU, each with six comma‑separated values:

        ``util, memory, temp, power, used_mem, total_mem``.
        """
        gpus: List[GPUInfo] = []
        for line in output.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 6:
                raise NvidiaSMIError(f"Unexpected nvidia-smi line: {line}")
            try:
                util, mem, temp, power, used, total = map(float, parts)
                gpus.append(
                    GPUInfo(gpu_util=util, vram_used=used, vram_total=total,
                            temp=temp, power=power)
                )
            except Exception as exc:  # pragma: no cover
                raise NvidiaSMIError(f"Failed to parse {line!r}: {exc}") from exc
        return gpus

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None