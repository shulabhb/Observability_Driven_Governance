# Enrichment summary

## Outputs

- `enriched_monitoring_report.xlsx` — enriched model sheets with **archetype-aware telemetry**; **risk tier / risk status columns are omitted** so you can train or rules-engine risk labels later from observables.
- Dashboard sheets are copied unchanged from `Monitoring Report.xlsx`.

## Telemetry archetypes

Each row includes `telemetry_archetype` (`tabular_ml`, `time_series`, `vision_cnn`, `vision_document`, `sequence_anomaly`, `nlp_classifier`, `speech_asr`, `reinforcement_learning`, `hybrid_rules_ml`, `llm_rag`, `llm_text`, `agentic_system`).

Only the block that matches the archetype is populated; others are `NaN` (e.g. LLM narrative fields for tabular models; document OCR fields for ViT KYC; WER proxy for speech).

## Omitted columns (feature-safe export)

**Legacy risk labels:** `risk_status`, `risk_tier`, `Risk Tier` — not written.

**Label proxies** (overlap intended predictions such as risk level, issue flags, human review, remediation action):  
`human_review_required`, `audit_readiness_score`, `sensitive_data_exposure_risk`, `evidence_linked_flag`, `concept_shift_flag`, `prompt_injection_flag`, `access_control_review_status`, `roi_pressure_flag`, `unresolved_audit_findings`, `remediation_cost_estimate_usd`, `reviewer_override_rate`, `customer_impact`, `compliance_incident_count_enriched` — not written. Use raw `compliance_incidents` where present.

## Internal stress shaping

Synthetic fields are scaled using **observable KPIs only** (drift, error, escalation, audit coverage, accuracy, compliance incidents), not using the removed risk labels.

## Row counts

- Pre-Go-Live: 50
- Post-Go-Live: 50
