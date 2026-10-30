# Architecture decision records

## ADR-007: Postgres as the primary datastore

We chose PostgreSQL 16 over a document database because our data is relational, our team knows SQL, and pgvector covers our vector search needs. Decision date: March 2026.

## ADR-012: Event bus

Internal services communicate through NATS JetStream. Kafka was rejected because of operational cost at our scale. Messages are JSON with a mandatory `schema_version` field.

## ADR-015: Feature flags

Feature flags are managed in Unleash. Flags older than 90 days without a change are reviewed for removal in the monthly architecture meeting.
