# VishGym Safety Boundary

VishGym is a closed defensive research simulator. It may model high-level risk decisions, but it never operates outside its container.

## Prohibited inputs and effects

- No real PII, contact data, bank details, UPI IDs, payment accounts, or credentials.
- No real-person voice references, public-video scraping, voice-upload endpoint, identity matching, or voice impersonation.
- No external browser navigation, web search, email/SMS/WhatsApp access, payment request, or exposed portal.
- No automatic publication or deployment of a newly trained adapter.

## Allowed synthetic surfaces

- `sandbox.local` pages backed by a fixed generated corpus.
- Synthetic voice output through Qwen3-TTS CustomVoice built-in timbres.
- Virtual wallet and pseudo-credentials marked `synthetic` in every event.
- Human-reviewed episode batches and adapters.

## Incident response

Treat any attempt to invoke an unapproved tool, navigate outside `sandbox.local`, use an external recipient, or access a voice reference as an invalid action. Record the invalid event, terminate the action, and exclude it from promotion data.
