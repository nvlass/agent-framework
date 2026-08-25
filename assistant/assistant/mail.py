"""Jailed outbound email for the assistant.

Recipients are a fixed alias -> address map from the YAML config. The agent
only ever names an alias; raw addresses never pass through the LLM. Delivery
goes through the local sendmail binary with a fixed argument list
(subprocess, shell=False), same approach as the news-agent digest.
"""

import re
import subprocess
from datetime import date

_MAX_BODY_CHARS = 50_000


class MailSender:
    def __init__(
        self,
        recipients: dict[str, str],
        from_addr: str,
        sendmail_path: str = "/usr/sbin/sendmail",
        subject_prefix: str = "",
        max_per_day: int = 20,
    ) -> None:
        if not recipients:
            raise ValueError("mail: at least one recipient alias is required")
        self._recipients = dict(recipients)
        self._from = from_addr
        self._sendmail = sendmail_path
        self._prefix = subject_prefix
        self._max_per_day = max_per_day
        self._sent_today = 0
        self._count_date = date.today()

    @property
    def aliases(self) -> list[str]:
        return sorted(self._recipients)

    def send(self, to: str, subject: str, body: str) -> str:
        addr = self._recipients.get(to)
        if addr is None:
            return (
                f"Error: unknown recipient {to!r}. "
                f"Allowed recipients: {', '.join(self.aliases)}"
            )

        today = date.today()
        if today != self._count_date:
            self._count_date = today
            self._sent_today = 0
        if self._sent_today >= self._max_per_day:
            return f"Error: daily email limit reached ({self._max_per_day}/day)"

        subject = re.sub(r"[\r\n]+", " ", subject).strip()
        if self._prefix and not subject.startswith(self._prefix):
            subject = f"{self._prefix} {subject}"
        if len(body) > _MAX_BODY_CHARS:
            body = body[:_MAX_BODY_CHARS] + "\n\n[... truncated ...]"

        message = f"To: {addr}\nFrom: {self._from}\nSubject: {subject}\n\n{body}\n"
        try:
            result = subprocess.run(
                [self._sendmail, "-t", "-i"],
                input=message.encode("utf-8"),
                capture_output=True,
                timeout=30,
            )
        except FileNotFoundError:
            return f"Error: sendmail not found at {self._sendmail}"
        except subprocess.TimeoutExpired:
            return "Error: sendmail timed out after 30s"
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace").strip()
            return f"Error: sendmail exited {result.returncode}: {err}"

        self._sent_today += 1
        return f"Email sent to {to} <{addr}>: {subject}"
