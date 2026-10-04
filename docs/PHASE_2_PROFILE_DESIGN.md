# Phase 2 design: family profiles and conversation continuity

**Status: Phase 2A synthetic UX prototype implemented and locally validated.**
The feature branch adds only the selector described here: two fixed fictional
labels, volatile browser state, a clear-session action, and a conversation reset
when switching. The selector is shown only for loopback development requests
with the offline answer engine; it is hidden on public/hosted deployments. There
is no profile API, database, account system, or personalized health behavior.

The design below remains the gate for any future persistent profiles. Do not
enter real names, dates of birth, medication lists, diagnoses, or individual lab
values.

Family MedGuard is an early educational prototype, not a clinical service, a
medical device, or a HIPAA-compliant system. This design is not legal, clinical,
privacy, or security approval for handling protected health information (PHI).

## Goal

Explore whether a person can keep a conversation organized for a selected
fictional household member without implying that the system can safely make
personalized medical decisions. The intended outcome of the first iteration is
only a usability prototype and a list of requirements—not a patient-record
system.

## Scope for a safe first prototype

- Use a fixed set of obviously fictional profiles such as **Demo Person A** and
  **Demo Person B**. Do not allow free-text profile creation.
- Show the selector only in local development on a loopback host with the
  offline answer engine; hide it when hosted or in production.
- Keep the selected demo profile and any demo conversation state in volatile
  memory only. Clear them on reset, tab close, or page refresh.
- Do not use `localStorage`, `sessionStorage`, cookies, a database, the vector
  index, analytics, or server-side chat-history storage for profile state.
- Do not put a profile identifier or profile data in model prompts, RAG queries,
  telemetry, or logs. The current generic chat behavior remains unchanged.
- Show a persistent banner: **Synthetic demo profile — never enter real health
  information.** Include a clear “Clear demo session” action.
- Do not collect or display real ages, birth dates, diagnoses, medicines,
  allergies, lab results, insurance data, addresses, or caregiver details.
- Keep the feature disabled in the public demo until the team explicitly
  approves the privacy and safety review described below.

This scope can test navigation and profile switching, but it is not true,
persistent multi-person health memory. That limitation must be stated plainly.

## Non-goals

- Diagnosis, treatment selection, dose calculation, medication-interaction
  checking, or interpretation of a person's laboratory results.
- Importing prescription labels or lab reports into a profile.
- Persistent profiles, saved conversations, caregiver accounts, or cross-device
  synchronization.
- EHR/FHIR connections, insurance workflows, appointment management, or
  clinician messaging.
- Any claim of HIPAA compliance, clinical validation, or regulatory clearance.

## Privacy and security boundary

The existing app has no authentication or authorization layer. The current
read-only configuration protects the bundled reference corpus; it does not
provide access control for private records. Therefore, adding persistent real
profiles to the current architecture would be unsafe.

Before considering real profile storage, obtain explicit product approval and
qualified privacy, security, clinical, and legal review. At minimum, define:

1. **Purpose and minimum data:** identify a concrete user need for every field;
   omit fields that are not necessary. Do not decide a health-profile schema by
   copying a medical record.
2. **Identity and authorization:** account lifecycle, strong authentication,
   session expiration, account recovery, household membership, roles, consent,
   and server-side authorization on every profile and conversation request.
   Hiding a control in the browser is not access control.
3. **Family and caregiver consent:** who can create, view, edit, export, and
   delete another person's information; how adult consent and minors' records
   are handled; how permission is revoked; and how shared-device risks are
   reduced.
4. **Data lifecycle:** encryption in transit and at rest, key management,
   retention limits, deletion from primary storage and backups, export,
   auditability, incident response, and recovery testing.
5. **Third parties:** keep health content out of hosted model, analytics, and
   error-reporting services unless each transfer has been reviewed and the
   necessary contractual, privacy, and security controls are in place.
6. **Safety and clinical oversight:** define which profile facts are allowed to
   affect an answer, what requires clinician review, how stale or conflicting
   information is surfaced, and how emergency and medication-safety handoffs
   remain effective. A profile must never silently weaken a safety handoff.
7. **Legal and regulatory review:** determine applicable obligations with
   qualified counsel. Do not describe the prototype as “HIPAA compliant” or
   “clinically safe” based on this design.

## Threats and required controls

| Threat | Example failure | Prototype control | Required before real data |
| --- | --- | --- | --- |
| Wrong person selected | A question is attributed to the wrong family member | Fixed fictional profiles; visible selected-profile banner; reset action | Confirm selection at consequential steps; test profile isolation end to end |
| Context leaks between people | One person's details appear in another person's conversation | No real facts; no profile fields sent to the model or retrieval | Tenant-scoped server authorization, isolation tests, and reviewed data flows |
| Shared device exposes information | A family member leaves a profile open | Synthetic data only; volatile state; clear-on-reset | Session locking/expiry, account controls, and privacy review |
| Logs or vendors retain content | A prompt or error report contains health details | No real data; no profile payload; current local echo model | Content-free logs, reviewed subprocessors, approved contracts, and retention controls |
| Stale or incorrect profile data | Old medicine or lab data is treated as current | No clinical fields in the prototype | Provenance, timestamps, user confirmation, clinician-directed correction, and safe handling of conflicts |
| Caregiver lacks authority | A user views or edits another adult's information without consent | No real household membership or sharing | Explicit authorization, consent records, revocation, and legal review |
| Overreliance | A profile makes an answer seem clinically personalized | Generic answers and existing safety gates remain unchanged | Human-factors testing, reviewed content, hazard analysis, monitoring, and regulatory assessment |

## Suggested implementation sequence

### Phase 2A — synthetic UX study

1. Design a static profile switcher using only fixed fictional labels.
2. Keep state in memory and reset it on refresh; do not add browser or server
   persistence.
3. Keep the profile identifier out of `/v1/chat`, the RAG index, prompts, and
   logs. The profile is a UI-only demonstration aid.
4. Test switching and clearing profiles, reload behavior, keyboard/accessibility
   controls, and the existing medical safety handoffs.
5. Run a usability review with synthetic scenarios only.

### Phase 2B — requirements and risk review

Document intended users, consent and caregiver roles, minimum necessary data,
retention/deletion, threat model, clinical hazards, vendor data flows, and legal
questions. No real data collection or persistent storage is permitted in this
phase.

### Phase 2C — implementation decision

Only after written approval from the project owner and qualified reviewers
should the team decide whether to build persistent profiles. If approval is not
obtained, keep the product at the synthetic, non-persistent prototype or remove
the profile feature.

## Acceptance criteria for Phase 2A

- Only fixed synthetic profile labels are available; there is no profile-entry
  form.
- The selector is hidden unless the app is in development, the client/host are
  loopback, and the provider is the offline echo engine.
- A visible warning identifies the profile as synthetic and forbids real health
  information.
- Profile selection is not persisted in local/session storage, cookies, the
  database, or the vector index; refreshing the page clears it.
- Network tests confirm no profile identifier or profile fields are sent to the
  chat, search, or OCR endpoints.
- Switching profiles clears any prior demo conversation state; no content from
  one demo profile appears under another.
- Existing diagnosis, emergency, personal-lab, dosage, and interaction handoffs
  remain unchanged and pass regression tests.
- No hosted model, analytics, or error-reporting service receives profile or
  conversation content.
- Documentation explicitly says the feature is a synthetic UX experiment and
  not suitable for real patient information.

## Open decisions before any real profiles

- Which user problem should profiles solve that cannot be solved by separate
  browser sessions?
- Is conversation continuity needed, or is a profile switcher alone sufficient
  for early usability research?
- Which household relationships, ages, caregiver roles, and consent rules are in
  scope, if any?
- What jurisdiction, data controller, hosting model, and retention policy would
  apply if real data were ever proposed?
- Who is accountable for clinical content, safety review, privacy, security, and
  incident response?

Until these questions and reviews are resolved, Phase 2 should remain a
synthetic-only design exercise—not a profile database or personal medical
assistant.
