# PROMPTS_FRONTEND.md: Shelfwatch Dashboard (Member 4)

**You own:** everything in `dashboard/`: `index.html`, `config.js`, `css/style.css`, `js/api.js`, `js/charts.js`, `js/app.js`, `dashboard/mock/*.json`.
**Visual reference:** `dashboard/prototype/shelfwatch_dashboard.html` (copy it there first). It uses the real UCI sales of 30 products and calculates risk in the browser; the production version keeps its look and takes all values from the API.
**API contract:** SRS section 7 and `CONTRACTS.md` C7. Member 3 builds the API; you start in **mock mode**.
**Setup first:** Phase 0 in `PROMPTS.md`. **Mode:** Planning. For the browser checks (F10) install Antigravity's browser extension so the agent can drive and screenshot the page.

**Non-negotiable rules** (already in `.agents/rules/04-frontend-rules.md`): plain HTML, CSS and JavaScript; no framework, no build step; all API access through `js/api.js` only; escape every piece of API text (`textContent` or an escape function, never raw `innerHTML`); light and dark themes through CSS variables; keyboard accessible; respect `prefers-reduced-motion`; total size under 150 KB excluding fonts; no business logic or risk calculation in the UI.

**Design tokens (put these in every relevant prompt or in the rule file):**

| Token | Light | Dark |
|---|---|---|
| `--bg` | #E4E9ED | #0F1A24 |
| `--pn` (panel) | #F5F7F8 | #16232F |
| `--ink` (text) | #16212B | #E8EEF3 |
| `--mu` (muted) | #566470 | #8FA0AF |
| `--ln` (lines) | #BFC9D1 | #2A3B4A |
| `--pl` (steel) | #2B3C4B | #4A6074 |
| `--bin` | #D3DAE0 | #1D2D3B |
| `--y` (accent) | #FFC72C | #FFCB3D |
| `--hi` / `--me` / `--lo` | #D93A2F / #E8791A / #1F8F5F | #FF6154 / #FF9A3C / #3DD598 |

Fonts: Big Shoulders Display (headline, numbers), Public Sans (text), from Google Fonts with fallbacks Arial Narrow, Impact, system-ui. Style: flat panels with 1 px lines, no shadows or glass, no gradients.

**Closing check (add to any prompt that lacks it):**
```
Before you finish: show how I can open and test this (exact commands or URL), list every file you created or changed, and list anything in the SRS sections I named that you did NOT implement. Do not claim success unless you actually loaded the page and checked it.
```

---

## Task F1: Page shell, theme and layout

**SRS:** 8.1, 8.4, FR-U-3, FR-U-9, FR-U-13 | **Output:** `index.html`, `css/style.css`

```
Read docs/SRS.md sections 8.1 to 8.4 and open dashboard/prototype/shelfwatch_dashboard.html.

Create dashboard/index.html and dashboard/css/style.css (no JavaScript logic yet, only a script tag for js/app.js) reproducing the prototype's shell exactly:
- header: logo (CSS-drawn shelf icon in yellow and steel) with the name "Shelfwatch", five nav tabs styled like shelf-edge labels (square, 6 px yellow left border; active tab filled steel), and a theme toggle button
- the hazard-tape disclaimer strip (diagonal yellow and black stripe on the left edge) containing the text from GET /disclaimer; put a static fallback with the same text in the HTML, and make js/app.js replace it from the API later
- the control rail: three range sliders with plain-language labels and live value outputs: "How long a delivery takes" (1 to 4 weeks, step 1, default 1), "Chance of not running out" (80 to 99 percent, default 95), "Starting stock, in weeks of sales" (0.5 to 8, step 0.5, default 3), plus a Reset button
- a main container, a footer note

Implement the tokens above as CSS variables on :root with dark values under @media (prefers-color-scheme: dark) guarded by :root:not([data-theme="light"]) and again under :root[data-theme="dark"]. Declare <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"> and apply safe-area padding to :root. Load fonts from Google Fonts with fallbacks. Add :focus-visible outlines (3 px yellow) and a prefers-reduced-motion rule disabling transitions.

Acceptance: the page renders in light and dark mode, works at 390 px width without horizontal scrolling, and text contrast is readable in both themes. Do not add pages yet.
```
**Done when:** opening `dashboard/index.html` shows the shell in both themes.

---

## Task F2: `config.js` and `api.js` (mock first)

**SRS:** FR-U-1, FR-U-5, FR-U-10, 8.4 behaviour rules | **Output:** the data layer

```
Read docs/SRS.md sections 7.1 and 8.4 and CONTRACTS.md C7.

1. dashboard/config.js: window.SHELFWATCH = { apiBase: "mock" } (allowed values: "mock", "" for same origin, or a full URL such as "https://my-api.onrender.com").
2. dashboard/mock/: create JSON fixtures that match the real API shapes exactly: products.json (30 products with product_id, description, total_units), forecast_<product_id>.json for each (history 52 weeks, test_predictions 8 weeks, next_week_forecast, closed_weeks), risk_items.json (items in contract C6 columns for the default parameters), metrics.json, error_analysis.json (segments, weeks, products), disclaimer.json. Derive history from the prototype's embedded data so the mock looks real.
3. dashboard/js/api.js: the ONLY module that talks to the API. Export getProducts(), getForecast(id), getRisk(params), getSensitivity(params), getMetrics(), getErrorAnalysis(), getDisclaimer(). In "mock" mode read the fixtures and, for getRisk and getSensitivity, compute results from the fixture forecast using the formulas in SRS section 6.2 ONLY inside a clearly marked mock adapter (function mockRisk) that is not used when apiBase is not "mock". In live mode call fetch(apiBase + path) with JSON bodies for POST.
4. Behaviour: debounce getRisk calls triggered by sliders by 150 ms; abort the previous in-flight request with AbortController when a newer one starts; cache results per parameter set (key = JSON of params); map HTTP 422 to the message "Those settings are out of range" and network or 5xx errors to "The server is not reachable. Check that the API is running." Return {ok, data, error} objects instead of throwing so the UI can render error states.
5. A small test page or script (dashboard/tests/api_smoke.html or a node script) that calls every function in mock mode and prints pass or fail.
```
**Done when:** the smoke script passes for all seven functions.

---

## Task F3: App core: routing, state and rendering

**SRS:** 8.4 (behaviour rules) | **Output:** `js/app.js` core

```
Create dashboard/js/app.js with a small framework-free structure:
- state: {L: 1, sl: 0.95, isw: 3, sort: "rank", dir: 1, filter: "ALL", selected: 0}
- hash routing with five routes: #overview, #explorer, #risk, #models, #sensitivity (default #overview); a views map of render functions; go() renders nav, then the current view; the nav marks the active tab with aria-current="page"
- slider wiring: updating state, the value outputs and calling an update function. On #overview update only the shelf bins in place (see F4) so the refill transition runs; on other pages re-render the view
- theme toggle: reads the system theme, flips data-theme on :root
- a helper el(tag, attrs, children) and esc(text) for building DOM safely; never assign API text to innerHTML
- a reusable renderState({loading, error, empty}, retryFn) that shows a loading line, an error panel with the message and a Retry button, or an empty message
- load /disclaimer once and put its text in the tape strip

Each view function returns a DOM node and receives data from api.js. Stub the five views with placeholder panels for now. Verify with the mock fixtures that navigation and sliders work and that no console errors appear.
```
**Done when:** navigation works and sliders update the outputs.

---

## Task F4: Shelf page (the signature page)

**SRS:** FR-U-11, FR-U-12 | **Output:** `views.overview`

```
Read docs/SRS.md FR-U-11 and FR-U-12 and the Shelf page in the prototype.

Implement the #overview view:
- a headline (h1, Big Shoulders Display, clamp(32px, 6.4vw, 64px)) that states the live count of HIGH-risk products: "In this simulation, N of M products run out before the next delivery." (singular: "runs out"; zero: "no product runs out"), and a subline: how many more are below their reorder point and how many have enough stock. Counts come from getRisk().summary, not from counting in the UI.
- the shelf wall: a CSS grid (repeat(auto-fill, minmax(54px, 1fr))), one bin per product ordered by total_units descending. Each bin is a <button> with a tank (150 px tall, steel border, bottom plank shadow) and a code label. Inside the tank: a fill whose height = on_hand / forecast_weekly weeks, clamped 0 to 6 weeks (percentage = weeks / 6); a red dashed line at forecast_lead_time / forecast_weekly; a yellow tick at reorder_point / forecast_weekly. Bin colour = risk_level (red, orange, green) through a CSS class AND the title text says the risk level in words ("high risk").
- the key (colour = risk; yellow tick = reorder point; red dashed line = demand while the delivery is on its way) and a one-line explanation.
- Motion: on first render the bins fill with a staggered transition (about 25 ms per bin); afterwards slider changes update the existing bins by setting CSS variables (--h, --lt, --rp) and the class, so the fill and lines animate smoothly with no re-render and no stagger. Respect prefers-reduced-motion.
- Clicking or pressing Enter on a bin sets state.selected and navigates to #explorer.
- For 100 bins the grid must stay readable: minimum bin width 44 px at narrow widths, wrapping into rows with a visible plank under each row.

Acceptance: bin count equals the product count; each bin's class equals risk_level; changing a slider changes heights smoothly; keyboard Tab reaches bins; works in both themes.
```
**Done when:** bins refill smoothly and the headline updates live.

---

## Task F5: Product forecast page and charts

**SRS:** FR-U-8 | **Output:** `views.explorer`, `js/charts.js`

```
Create dashboard/js/charts.js with hand-built SVG helpers (no chart library): lineChart({series, xTicks, yTicks, bands, markers}), bars(values, labels), and a tooltip helper. Then implement #explorer:

- a product <select> (code: description; ordered like the shelf) and a risk badge (icon, colour and text) from getRisk()
- a line chart from getForecast(id): actual weekly units for all 52 weeks (solid ink line); a shaded band for the 8 test weeks labelled "test weeks"; a thin red band on the closed week labelled "shop closed"; the model's predictions for the test weeks (dashed steel line); the next-week forecast for 2011-12-05 as a yellow square marker at the right edge; y gridlines with thousand separators; x ticks every 8 weeks with short dates; a caption and axis labels with units (units per week)
- hover: a vertical crosshair and a tooltip showing the date, actual and prediction (or "forecast" for the last point); keyboard users can step through weeks with the left and right arrow keys when the chart is focused (aria-label describes the chart)
- six stat tiles: average units a week, forecast for the next week, typical forecast miss (sigma_error from risk items or forecast payload), simulated stock, reorder point, days of cover
- loading, empty and error states via renderState.

Acceptance: switching products updates everything; the closed week and test band are visible; no console errors; charts have titles and units.
```

---

## Task F6: Risk table

**SRS:** FR-U-6, FR-U-7

```
Implement #risk:
- filter buttons "All N", "High n", "Medium n", "Low n" (counts from the summary; icon plus text for each level)
- a table with columns Code, Product, Forecast a week, Simulated stock, Safety stock, Reorder point, Days of cover, Risk. Header cells are buttons: click sorts by that column, click again reverses; the sorted column shows an arrow and aria-sort. Default sort: risk level (high first), then days of cover ascending.
- numeric cells use tabular figures and thousand separators; days of cover shows a dash when 999
- the Risk cell shows icon, colour and the word (HIGH, MEDIUM, LOW); never colour only
- row click or Enter opens that product in #explorer
- "Download CSV" exports the rows currently shown (respecting filter and sort) with the contract C6 column names; build it with a Blob and a temporary link
- the table scrolls horizontally inside its own container on narrow screens (the page body must not scroll sideways)

Acceptance: sorting and filtering work from the keyboard; CSV opens correctly in a spreadsheet; counts on buttons equal the summary.
```

---

## Task F7: Model performance page

**SRS:** 8.2 | **Output:** `views.models`

```
Implement #models using getMetrics() and getErrorAnalysis():
- a metrics table for every model: MAE, RMSE, WAPE (as percent), bias (signed), with a split switch (validation / test) and the best model marked
- a "how to read this" paragraph in plain words (lower is better; bias near zero is better; WAPE is total error divided by total sales)
- bars of WAPE per test week for the best model (use bars() from charts.js) with week labels
- error by volume segment (low, medium, high) as a small grouped table or bars
- an over-forecast versus under-forecast summary (units) with one sentence: "Under-forecasting leads to stockouts."
- rows for models not yet available are not shown (do not invent placeholders); if /metrics fails show the error state

Acceptance: numbers on screen equal the API payload; the best model is highlighted with text, not only colour.
```

---

## Task F8: Sensitivity page

**SRS:** 8.2

```
Implement #sensitivity using getSensitivity({lead_times: [1,2,3,4], service_levels: [0.90, 0.95, 0.99]}) with the current "starting stock" slider value:
- a heatmap grid (4 lead-time rows by 3 service-level columns); each cell shows the HIGH count (large number) and the MEDIUM count (small), background intensity proportional to (HIGH + MEDIUM) / total, text always readable; the cell matching the current slider settings is outlined in yellow
- a short paragraph explaining how to read it
- an "Assumptions" panel listing in plain words: stock is simulated (3 weeks of average sales by default times a fixed random factor between 0.5 and 1.5 per product, seed 42); demand during delivery = weekly forecast x delivery weeks; safety stock = z x typical forecast miss x square root of delivery weeks; reorder point = demand during delivery + safety stock; HIGH / MEDIUM / LOW rules; days of cover = stock / (forecast / 7)
- re-fetch when the starting-stock slider changes

Acceptance: the outlined cell tracks the sliders; cell numbers equal the API response.
```

---

## Task F9: States, accessibility and motion

**SRS:** FR-U-4, FR-U-5, FR-U-13, FR-U-14

```
Audit and fix the whole dashboard for:
1. Loading, empty and error states on every page; with the API unreachable every page shows the message and a Retry button, never a blank page or an uncaught exception.
2. Risk is never conveyed by colour alone (icon + word everywhere: shelf title text, badges, table, heatmap).
3. Keyboard: every control reachable in a logical Tab order with a visible 3 px yellow focus ring; bins, table headers, filter buttons, chart (arrow keys), theme toggle and sliders operable without a mouse; skip-to-content link; aria-current on the active tab; aria-live="polite" on the headline so live changes are announced.
4. Contrast: check text and key UI colours against WCAG AA in both themes and fix failures by adjusting tokens (report old and new values).
5. prefers-reduced-motion disables fills, staggers and any pulsing.
List every issue found and every fix.
```

---

## Task F10: Browser verification (use the agent's browser)

**SRS:** T-U-1 to T-U-4 | **Mode:** Planning

```
Serve dashboard/ in mock mode (python -m http.server 5173 --directory dashboard) and, using the browser, do all of the following, taking a screenshot of each page in light and dark mode:
1. Visit #overview, #explorer, #risk, #models, #sensitivity.
2. Move each slider to its minimum, default and maximum; confirm the shelf refills and the headline and heatmap update.
3. Toggle the theme; reload and confirm the system theme is respected.
4. Using only the keyboard, open a product from the shelf, sort the risk table, download the CSV.
5. Resize to about 390 px wide and 820 px wide; confirm nothing overflows horizontally and the shelf wraps cleanly.
6. Open the console; confirm no errors or warnings.
7. Stop the server or set apiBase to a wrong URL and confirm every page shows the friendly error with Retry.
Report a table of checks (pass or fail with evidence) and fix every failure. Do not change the visual design.
```
**Done when:** every check passes and screenshots are attached.

---

## Task F11: Size, performance and security audit

**SRS:** NFR-1, NFR-8, NFR-12

```
1. Measure the size of index.html, style.css, app.js, api.js, charts.js and the total excluding fonts and mock data. It must be under 150 KB uncompressed; if not, find and remove dead code.
2. Search all JS for innerHTML, outerHTML, insertAdjacentHTML, eval, new Function, document.write and inline event handler attributes; justify or remove each. Every string from the API must reach the DOM through textContent or esc().
3. Confirm there are no network requests except the API and Google Fonts (check the network panel), no localStorage use that stores anything sensitive, and no console logging of full API payloads.
4. Measure page load and slider-to-update time on the mock data and report milliseconds.
```
**Done when:** a short audit table lists each check, result and fix.

---

## Task F12: Switch to the live API

**Depends on:** Member 3 finishing the API (B6 to B11)

```
Set dashboard/config.js to apiBase "" (same origin) and run: uvicorn api.main:app --port 8000, then open http://localhost:8000/. Remove the mock risk adapter (mockRisk) from api.js entirely; the UI must use ONLY values from POST /risk and POST /sensitivity. Compare every page against the API payloads (use the browser network panel): headline counts equal summary.n_high; bin classes equal risk_level; forecast chart equals /forecast/{id}; model performance equals /metrics. Fix mismatches in the UI (field names, nulls, ordering). If the API must change, add a line to CHANGE_REQUESTS.md and tell me; do not edit the API yourself. Keep mock mode available through config.js for offline work.
```
**Done when:** the dashboard works at `http://localhost:8000/` with real numbers. Commit: `feat(dashboard): Shelfwatch live`.
