# PROMPTS_FRONTEND.md: Shelfwatch Multi-Page Web App (Member 4), version 3

**You own:** everything in `dashboard/`: five pages (`index.html`, `predict.html`, `intelligence.html`, `inventory.html`, `analysis.html`), `config.js`, `css/style.css`, `js/*.js`, `dashboard/mock/*.json`, and `docs/JS_GUIDE.md`.
**Visual reference:** `dashboard/prototype/` (the single-page Shelfwatch prototype). Keep its look; **split it into five real pages** (mentor: not a single-page app).
**API:** SRS section 7 and `CONTRACTS_ADDENDUM.md` (all endpoints under `/api`). Start in **mock mode**; switch to live in F11.
**Primary user:** the Retail Inventory and Replenishment Manager (SRS 1.6, `docs/STAKEHOLDER.md`). Every page supports one step of their weekly routine; show that step as a one-line subtitle under each page title (FR-U-22). Design for this one person only: no role switchers, no screens for other stakeholders.
**Setup:** paste the Windows preface and closing check from `PROMPTS_00_INDEX.md`. Planning mode. Install Antigravity's browser extension for F9.

**Non-negotiable rules:** plain HTML, CSS and JavaScript (no framework, no build step); all API access through `js/api.js`; every API string reaches the DOM through `textContent` or an escape function (never raw `innerHTML`); no `eval`; no inline event attributes (use `addEventListener`); light and dark themes through CSS variables; keyboard accessible; respect `prefers-reduced-motion`; total size under 150 KB excluding fonts; no model or risk logic in the UI.

**Design tokens**

| Token | Light | Dark |
|---|---|---|
| `--bg` | #E4E9ED | #0F1A24 |
| `--pn` (panel) | #F5F7F8 | #16232F |
| `--ink` | #16212B | #E8EEF3 |
| `--mu` (muted) | #566470 | #8FA0AF |
| `--ln` (lines) | #BFC9D1 | #2A3B4A |
| `--pl` (steel) | #2B3C4B | #4A6074 |
| `--bin` | #D3DAE0 | #1D2D3B |
| `--y` (accent) | #FFC72C | #FFCB3D |
| `--hi`, `--me`, `--lo` | #D93A2F, #E8791A, #1F8F5F | #FF6154, #FF9A3C, #3DD598 |

Fonts: Big Shoulders Display (headline, numbers), Public Sans (text), from Google Fonts with fallbacks Arial Narrow, Impact, system-ui. Style: flat panels with 1 px lines; no shadows, glass or gradients.

---

## Task F1: Shared shell for five pages
**SRS:** 8.1, FR-U-2, FR-U-3, FR-U-13, FR-U-17

```
Read docs/SRS.md sections 8.1 to 8.4 and open the prototype.

Create dashboard/css/style.css (tokens, light and dark through :root, @media (prefers-color-scheme: dark) guarded with :root:not([data-theme="light"]), and :root[data-theme="dark"]; safe-area padding; focus-visible 3 px yellow outline; reduced-motion rule), dashboard/config.js (window.SHELFWATCH = { apiBase: "mock", apiKey: "demo-key" }) and dashboard/js/common.js, plus the five empty page files. Every page has the same: header with the logo and a nav of five real links (Home, Predict, Product intelligence, Inventory manager, Analysis board) with aria-current on the active one; the hazard-tape disclaimer strip whose text is loaded from /api/disclaimer (with a static fallback of the same wording); a footer with links to the Kaggle model, notebook and dataset (from /api/model-info) and to /docs; a theme toggle that stores the choice in memory only.

common.js exports helpers on window.Shelf: el(tag, attrs, children), esc(text), formatNumber, formatDate, setActiveNav, renderState({loading, error, empty}, retryFn). No page logic here.
Acceptance: five pages open from each other with plain links, look the same in light and dark, and have no horizontal scroll at 390 px width.
```

## Task F2: `api.js` and mock fixtures
**SRS:** FR-U-1, FR-U-5, FR-U-10

```
Create dashboard/js/api.js: the ONLY module that talks to the API. Export getHealth, getDisclaimer, getModelInfo, getProducts, getForecast(id), predict(body), predictCustom(body), getRisk(params), simulate(params), getSensitivity(params), getMetrics, getErrorAnalysis, getIntelligence, getIntelligenceFor(id), getInventory, putInventory(id, body), deleteInventory(id), getOrders(filters), createOrder(body), patchOrder(id, body), deleteOrder(id), getLogs(limit, offset).

Behaviour: apiBase "mock" reads dashboard/mock/*.json fixtures; "" means same origin; or a full URL. In mock mode the mock adapter may compute risk from the fixture using the SRS formulas (function mockRisk) but it must NOT be used when apiBase is not "mock". Write requests send header X-API-Key from config.js. Use fetch with async/await, response.ok checks and try/catch; return {ok, data, error} objects (never throw to the page). Map 401 to "Missing or wrong API key", 409 to the server's message, 422 to "Those values are out of range", 429 to "Too many requests, wait a moment", and network or 5xx to "The server is not reachable". Debounce getRisk and simulate calls from sliders by 150 ms, abort the previous in-flight request with AbortController, cache results per parameter set and clear the cache after any write.

Create dashboard/mock/ fixtures matching CONTRACTS_ADDENDUM.md exactly for every endpoint (30 products from the prototype data are enough; include one null history week and two inventory records and three orders). Add dashboard/tests/api_smoke.html that calls every function in mock mode and prints PASS or FAIL.
```
**Done when:** the smoke page shows all PASS.

## Task F3: Home and Shelf (`index.html`)
**SRS:** FR-U-11, FR-U-12, FR-U-18

```
Implement index.html and js/home.js:
- control rail with three sliders (How long a delivery takes 1 to 4 weeks; Chance of not running out 80 to 99 percent; Starting stock in weeks of sales 0.5 to 8) and a Reset button; use input events with the debounce from api.js.
- headline "In this simulation, N of M products run out before the next delivery." (singular: runs out; zero: no product runs out) from getRisk().summary.n_high; a subline with the MEDIUM and LOW counts and total revenue at risk.
- the shelf wall: CSS grid repeat(auto-fill, minmax(54px, 1fr)), one button per product ordered by total_units; fill = on_hand / forecast_weekly weeks clamped 0 to 6; red dashed line at forecast_lead_time / forecast_weekly; yellow tick at reorder_point / forecast_weekly; colour AND icon AND title text for the risk class; a small "your stock" marker on bins whose on_hand_source is "user". First render fills with a staggered transition; afterwards slider changes update existing bins by setting CSS variables (--h, --lt, --rp) and the class so they animate smoothly. Use ONE click listener on the container (event delegation); selecting a bin or pressing Enter opens predict.html?product=<id>.
- the ACTION LIST directly under the headline (FR-U-21): groups ORDER_NOW, ORDER_SOON, REVIEW_STOP and HOLD built from the `action` field in the risk items (do NOT compute actions in the browser). Each row: product code and name, recommended quantity, days to reorder, revenue at risk, and a link to predict.html?product=<id>. Show the count in each group header, order rows by urgency (revenue at risk descending inside a group), and collapse HOLD by default. Use an aria-live region so the list announces changes when sliders move.
- below the wall, static sections (keep them short; the manager is the audience): "Who this is for" (the Retail Inventory and Replenishment Manager: one short persona line and their weekly routine from SRS 1.6), "Where it is used" (SRS 1.6 real-life paragraph), "How it works" (4 short steps), and links to the live API docs and the three Kaggle pages.
Acceptance: bin count equals product count; each bin class equals risk_level; sliders animate smoothly; keyboard reaches every bin; works in both themes.
```

## Task F4: Predict page (`predict.html`)
**SRS:** FR-U-15, FR-U-8

```
Implement predict.html and js/predict.js with two modes (tabs):
1. "Choose a product": a select (code and description), optional overrides (stock on hand, lead time, service level) and a Predict button. If the URL has ?product=<id> (read with URLSearchParams) preselect it and run automatically.
2. "Enter your own sales": a form with 8 to 12 weekly unit values (one input per week plus Add week and Remove week buttons), optional price, and a target date. Validate in the browser BEFORE sending: 8 to 12 numbers, none negative, valid date; show errors next to the field. Handle submit, input and change events.

Results (cards): forecast with the low-to-high range, a small SVG chart showing the recent weeks, the forecast point and the p10 to p90 band, the risk badge (icon, colour, word), the action label (Order now, Order soon, Review to stop, Hold), days of cover, stockout probability, recommended order quantity, days to reorder, revenue at risk, and a plain-words sentence such as "Order about X units now" or "No order needed this week". For custom input show the notes about unavailable feature groups.

What-if simulator panel: sliders for lead time (1 to 4), delay shock (0 to 2 weeks), service level and starting stock calling simulate() on input events (debounced); show the new HIGH, MEDIUM, LOW counts next to the current ones and the list of products whose class changed.
Acceptance: both modes work in mock mode with no console errors; invalid input is blocked with a message; results update without page reload.
```

## Task F5: Product intelligence (`intelligence.html`)
**SRS:** FR-U-19

```
Implement intelligence.html and js/intelligence.js using getIntelligence() and getIntelligenceFor(id):
- a search box (code or description), filter chips for demand pattern, ABC class and lifecycle stage, and a sortable table: code, product, pattern, ABC, XYZ, lifecycle, return rate, export share, revenue share.
- a distribution strip at the top showing the real class counts (do not invent categories).
- selecting a row opens a detail panel with the profile in plain words and the bought-together companions (confidence and lift with a one-line explanation of lift). Empty state if there are no companions.
- each row has a link to predict.html?product=<id>.
Acceptance: filters and sorting work from the keyboard; only real values are shown; empty states handled.
```

## Task F6: Inventory manager (`inventory.html`)
**SRS:** FR-U-16, FR-DB

```
Implement inventory.html and js/inventory.js. Two panels:
1. Stock: a table of products with forecast, risk badge, action and the stock source (user or simulated). Row action Edit stock opens an inline form (on hand, lead time, service level) that calls putInventory(); Reset to simulated calls deleteInventory() after a confirm dialog.
2. Purchase orders: a Create order form (product select, quantity, note) calling createOrder(); a table of orders with status badges and action buttons: Approve (Pending to Approved), Receive (Approved to Received), Cancel, Delete (only for Pending or Cancelled). Buttons that are not allowed for the current status are disabled with a title explaining why.

After EVERY successful write, re-fetch the risk data and update the affected rows and badges in place (no page reload) and show a short status message in an aria-live region. Show friendly messages for 401, 409, 422 and network errors. Confirm before every delete. Use event delegation for the table buttons. Add Export orders to CSV.
Acceptance: the full create, approve, receive, delete flow works in mock mode and against the live API later; the row badge changes when stock changes.
```

## Task F7: Analysis board (`analysis.html`)
**SRS:** FR-U-20, FR-M-17, 8.x

```
Implement analysis.html and js/analysis.js (nav label "Analysis") with a left section list and these 8 sections; each card ends with a small caption "Source: <MySQL tables>" taken from the API `source` field. Show only what the API returns; missing parts get a clear empty state, never placeholder numbers.
1. Data and lineage (getLineage): the two datasets with their roles and row counts, the cleaning funnel as horizontal bars (rows remaining per step), the train, validation and test timeline, and the overlap check result.
2. Features (getFeatureAnalysis): feature groups, importance bars coloured by group, and the ablation table (some groups may not help; say so).
3. Tuning (getTuning): per-candidate trial scatter or strip of validation WAPE, best trial marked with text, best parameters.
4. Evaluation (getEvaluation + getMetrics): metrics table with validation or test switch, WAPE as percent, signed bias, WAPE per test week, error by volume segment and pattern class, interval coverage versus the 80 percent target, over versus under-forecast units with the sentence "Under-forecasting leads to stockouts."
5. Business backtest (getBacktest): policy, lead time, fill rate, stockout weeks, average on-hand; a product picker that replays stock week by week (on-hand line, orders, unmet demand).
6. Explain a forecast: product and target week pickers calling getExplain; show the 12 weeks of history used, each feature's value and signed contribution as a waterfall (base value to model output), calendar and prior-year context, and the p10 to p90 interval.
7. How to get an output: a four-step diagram (choose product, features built, model run, risk rule) with the real values from the last prediction.
8. System log: request log table (time, method, path, status, duration ms) with Refresh and auto-refresh toggle, plus the table row counts from /api/health; one sentence that the middleware writes these rows.
Add the corresponding functions to api.js and mock files. No login anywhere; every request sends the header ngrok-skip-browser-warning: true.
```

## Task F8: States, accessibility and motion
```
Audit all five pages for: loading, empty and error states with a Retry button; risk never shown by colour alone; keyboard order and visible focus; a skip-to-content link; aria-current, aria-live on the headline and the status messages, aria-sort on sortable headers; WCAG AA contrast in both themes (report old and new token values if you change any); prefers-reduced-motion disabling fills, staggers and pulses. List each issue and each fix.
```

## Task F9: Browser verification (use the agent's browser)
```
Serve dashboard/ in mock mode (python -m http.server 5173 --directory dashboard). In the browser, for EACH of the five pages: open it from the nav, take screenshots in light and dark mode, check the console for errors, and test keyboard-only use. Then: move every slider to its minimum, default and maximum on Home and Predict; run both Predict modes including invalid input; do the full create, approve, receive, delete order flow; edit and reset a stock record; open the Analysis board and refresh the log; resize to about 390 px and 820 px wide and confirm no horizontal overflow; stop the server (or set a wrong apiBase) and confirm each page shows the friendly error with Retry. Report a table of checks with pass or fail and evidence, fix every failure, and do not change the visual design.
```

## Task F10: Size and security audit
```
1. Report the size of each HTML, CSS and JS file and the total excluding fonts and mock data; it must be under 150 KB. Remove dead code if not.
2. Search all JS for innerHTML, outerHTML, insertAdjacentHTML, eval, new Function, document.write and inline event attributes; justify or remove each.
3. Confirm every API string reaches the DOM through textContent or esc(); the API key is only sent on write requests; nothing sensitive is logged to the console.
4. Check network requests: only the API and Google Fonts. Measure slider-to-update time in milliseconds.
```

## Task F11: Switch to the live API
**Needs:** Member 3's B9 to B16

```
Set config.js to apiBase "" and open http://localhost:8000/ with uvicorn api.main:app running. Remove nothing from mock mode except make sure mockRisk is never called when apiBase is not "mock". Compare each page with the network responses: headline counts equal summary.n_high, bin classes equal risk_level, the forecast chart equals /api/forecast/{id}, orders and stock persist through reloads, the log table shows your own requests. Fix mismatches in the UI. If the API must change, add a line to CHANGE_REQUESTS.md and tell me; do not edit the API.
```
**Done when:** all five pages work against the real API.

## Task F12: JavaScript guide for the viva
**SRS:** 8.5

```
Create docs/JS_GUIDE.md from OUR code. For each concept in SRS 8.5 (event listeners, fetch with async and await, JSON, try and catch and response.ok, DOM updates and textContent, event delegation, debouncing, AbortController, URLSearchParams, CSS variables from JavaScript) write: a 3-sentence plain explanation, the file and function where it is used, a 10-line excerpt copied from the real code, and a viva question with a 2-sentence model answer. Add a final section "Trace one click": follow a click on a shelf bin from the event listener through the URL change, page load, fetch call, response handling and DOM update, naming the function at each step.
```

---

## Task F13: Plan workflow, explain panel and no-login check (added in version 3.2)
**SRS:** FR-W-1 to FR-W-6, FR-U-20
```
1. inventory.html: add a "Weekly plan" panel: Generate plan (calls POST /api/plans/generate), a table of order lines (product, quantity, action), Approve and Cancel buttons, Export CSV link, and a plan history list; after approval show the created orders and a Receive button per order that updates stock. Show 409 and 401 messages in plain words.
2. predict.html: add "How this forecast was made" (collapsible): the 12 history weeks used, top contributing features from getExplain, the p10 to p90 range, a link "Open in Analysis board" with the product preselected, a prediction history table (getPredictionHistory), and a "Show underlying transactions" drill-down for a chosen week (getTransactions, max 200 rows, no customer fields).
3. Confirm there is no login page, no sign-in form and no stored credentials; the site opens on index.html. The API key stays out of the browser code: writes go through the same-origin server which adds it, or the key is read from config.js documented as a demo key.
4. Every fetch sends header ngrok-skip-browser-warning: true. Add a small "Live / Mock" badge in the nav reflecting config.js.
Verify in the browser at both the local and the ngrok URL; screenshots in docs/screens/.
```
