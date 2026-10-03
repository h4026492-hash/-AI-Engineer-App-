# API Guide

The Lumen API is a JSON REST API at https://api.lumen.example/v1. All requests
require an `Authorization: Bearer <token>` header.

## Rate limits

Rate limits are per API token and use a fixed 60-second window.

| Plan       | Requests per minute | Burst |
| ---------- | ------------------- | ----- |
| Free       | 60                  | 10    |
| Pro        | 600                 | 50    |
| Business   | 3000                | 200   |
| Enterprise | Custom              | Custom |

Rate limit state is returned on every response in the `X-RateLimit-Limit`,
`X-RateLimit-Remaining`, and `X-RateLimit-Reset` headers. When you exceed the
limit the API returns HTTP 429 with a `Retry-After` header in seconds.

## Pagination

List endpoints are cursor paginated. Pass `limit` (max 100) and `cursor`. The
response includes `next_cursor`, which is null when there are no more pages.
Offset pagination is not supported and will not be added, because it is unsafe
while rows are being inserted.

## Errors

Errors return a JSON body of the form
`{"error": {"code": "string", "message": "string"}}` along with an appropriate
HTTP status. Codes are stable and safe to branch on; messages are for humans
and may change wording at any time.

## Idempotency

POST endpoints accept an `Idempotency-Key` header. Replaying a request with the
same key within 24 hours returns the original response instead of creating a
second resource.

## Webhooks

Webhooks are signed with HMAC-SHA256 using your endpoint secret. Verify the
`X-Lumen-Signature` header before processing. Delivery is retried with
exponential backoff for up to 72 hours, and events are delivered at least once,
so handlers must be idempotent.

## Versioning

The API version lives in the URL path. Breaking changes are only shipped in a
new major version. Non-breaking additions, such as new fields, may appear at
any time, so clients must ignore unknown fields.
