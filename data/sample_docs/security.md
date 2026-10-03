# Security and Data Handling

## Encryption

All data in transit is encrypted with TLS 1.3. Data at rest is encrypted with
AES-256. Encryption keys are held in a managed hardware security module and
rotated every 90 days.

## Authentication

Single sign-on via SAML 2.0 and OIDC is available on the Business and
Enterprise plans. SCIM provisioning is available on Enterprise only.

Passwords must be at least 12 characters long. We check new passwords against
a breach corpus at sign-up and at change time. Hardware security keys and
TOTP are supported for all users.

## Data residency

Customer data is stored in the region you select at workspace creation:
us-east, eu-central, or ap-southeast. Data does not leave the selected region,
including backups and logs. Changing region requires a migration performed by
support and causes up to 4 hours of read-only downtime.

## Subprocessors

We use a cloud provider for compute and storage, and a transactional email
provider for notifications. The current list is published at
lumen.example/subprocessors, and customers are notified 30 days before a new
subprocessor is added.

## Incident response

Security incidents are triaged within 30 minutes, 24 hours a day. Customers on
Business and Enterprise plans are notified within 24 hours of a confirmed
incident affecting their data. Annual penetration test summaries are available
under NDA.
