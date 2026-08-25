#!/bin/bash
# Send an email digest via sendmail. Recipient is fixed here (not agent-
# controlled) for security — the agent can only set subject and body.
# Override the addresses via env, or edit the defaults below.
RECIPIENT="${NEWS_EMAIL_TO:-you@example.com}"
FROM="${NEWS_EMAIL_FROM:-news-agent@localhost}"
SUBJECT="$1"
BODY="$2"

/usr/sbin/sendmail -t <<EOF
To: $RECIPIENT
Subject: $SUBJECT
From: $FROM

$BODY
EOF
