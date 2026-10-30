# Deployment runbook

## Environments

There are three environments: dev, staging and production. Staging mirrors production infrastructure and is refreshed from a production snapshot every Sunday at 03:00 UTC.

## Release process

Releases go out on Tuesdays and Thursdays. A release requires a green CI run, an approved pull request from a code owner, and a changelog entry. Hotfixes may be deployed any day with two approvals.

## Rollback

Rollback is performed with `deployctl rollback <service> --to <previous-tag>`. The previous two releases are always kept warm. A rollback must be announced in #deployments and followed by an incident note within 24 hours.

## On-call

The on-call engineer rotates weekly on Monday at 10:00. The escalation path is on-call engineer, then the service owner, then the head of engineering. Pager acknowledgement is expected within 15 minutes.
