---
trigger: model_decision
description: "Frontend rules relevant when editing or creating files in dashboard/."
---

# Frontend Rules (`dashboard/`)

- Use plain HTML, CSS, and JavaScript only; no framework and no build step.
- Route all API requests exclusively through `dashboard/js/api.js`.
- Escape all API-derived text (`textContent` or an explicit escape helper); never insert raw API strings via `innerHTML`.
- Support light and dark themes using CSS custom properties (variables).
- Ensure full keyboard accessibility across interactive controls.
- Respect `prefers-reduced-motion`.
- Keep total dashboard asset size under 150 KB excluding fonts.

