# EVALUATION_PLAN.md: run sheet for a 2-hour-plus evaluation

SRS 11.7. Several faculty will evaluate for more than two hours, so every part has a presenter, a screen, a time and a fallback. Rehearse it twice with a timer.

## 1. Roles
| Member | Module | Demonstrates |
|---|---|---|
| M1 | Data, enrichment, intelligence | Two datasets, cleaning funnel, product intelligence |
| M2 | Features and models | Split, features, tuning trials, ablation, intervals, explanation |
| M3 | Risk engine, MySQL, API | Backtest, MySQL tables, CRUD, middleware, Postman |
| M4 | Web pages, deployment | Live walk-through, JavaScript, ngrok, Kaggle |
One member **drives the browser** (M4). The presenter of each part narrates.

## 2. Timed run sheet (150 minutes)
| Time | Min | Part | Presenter | Screen | Fallback |
|---|---|---|---|---|---|
| 0:00 | 5 | Problem, the one stakeholder (Retail Inventory and Replenishment Manager), 2-sentence pitch | M4 | Slide 1 to 3 | Slides PDF |
| 0:05 | 10 | Data: two datasets, cleaning funnel (541,909 to 5,067 weekly rows), closure week, zero-fill decision | M1 | Analysis board section 1 | Screenshots |
| 0:15 | 10 | How data is split and why not random; leakage test; feature groups | M2 | Analysis board sections 2 and 3 | Notebook |
| 0:25 | 15 | Models and tuning trials; ablation (what helped and what did not); intervals and coverage | M2 | Analysis board sections 4 and 5 | Notebook |
| 0:40 | 10 | Business backtest: would following the advice have helped? Replay for one product | M3 | Analysis board section 6 | Backtest CSV |
| 0:50 | 20 | **Live system as the manager's week:** Home headline and Action list, open a product, Predict and "How this forecast was made", what-if, generate plan, approve, receive, Home updates | M4 drives, M3 narrates | Live site | Local address, screen recording |
| 1:10 | 10 | MySQL tour in Workbench: tables, row counts, system tables after the live demo; middleware and audit log | M3 | Workbench + Analysis board section 8 | Dump on second laptop |
| 1:20 | 10 | API: `/docs`, Postman collection on the local and the public address | M3 and M4 | Postman | Saved runner screenshots |
| 1:30 | 10 | JavaScript: trace one click on a bin through the event listener, `fetch`, response, DOM | M4 | VS Code + browser tools | `docs/JS_GUIDE.md` |
| 1:40 | 10 | Kaggle dataset, notebook, model; limitations and honesty (persona assumed, simulated stock, one-year history) | M1 and M2 | Kaggle pages | Screenshots |
| 1:50 | 40 | Open questions and faculty-driven exploration | All | Anything | Buffer |

## 3. Pre-flight (30 minutes before)
- [ ] Power connected; sleep and screen lock disabled; Wi-Fi stable (hotspot ready)
- [ ] MySQL running; `python scripts\reset_demo.py` run; row counts checked in `/api/health`
- [ ] `scripts\start_server.bat` running; `http://localhost:8000/` opens Home
- [ ] `scripts\start_ngrok.bat` running; public address opened on a phone (click Visit Site once)
- [ ] ngrok usage page checked
- [ ] Postman: public environment selected; collection run green; screenshots saved
- [ ] Browser tabs prepared in order; zoom 110 to 125 percent; dark and light theme both tested
- [ ] MySQL Workbench connected; the key tables open
- [ ] Second laptop: project copied, dump restored, server starts
- [ ] Slides PDF, report PDF, screen recording on a USB drive and in the cloud
- [ ] Each member has their notes and knows the other parts at a high level

## 4. Questions faculty may ask (prepare 2-sentence answers)
**Problem and stakeholder:** Why this stakeholder? What does their week look like? What is assumed?
**Data:** Why two datasets? What does Online Retail II add and what does it not? Why was the closure week removed and not set to zero? Why zero-fill after the last sale? What was cleaned and why? How big is the data?
**Split and leakage:** Why not a random split? What is shift-before-rolling? How do you know there is no leakage? Why is the test period the last 8 weeks?
**Features:** What is each group? Which did not help? Why is `guest_share` only a hypothesis? Why are weather and retail-index data not used?
**Models and tuning:** What did you compare? How was tuning done and why time-series cross-validation? Why did the moving average win before tuning? What is WAPE and why not MAPE? What is bias?
**Intervals and risk:** What does the p10 to p90 range mean? What was the measured coverage? How is safety stock computed? What is stockout probability and what does it assume? Why is the stock simulated and what changes when the manager enters real stock?
**Backend and database:** Why these MySQL tables? What does each CRUD call return? What is each middleware doing? Why an API key and not a login? How does Python call this API?
**Frontend:** How does a slider change the screen? What does `await` do? How is user input validated? Why five pages and not one?
**Deployment:** How does the public URL reach your laptop? What if the laptop sleeps? How was it tested?
**Honesty:** What does the system not do? Where can it be wrong? What would you do with more time?

## 5. Rules for the session
- Answer from the Analysis board and the data whenever possible: show, do not just say.
- If something fails, say what it is, switch to the fallback within 20 seconds, and continue. Never debug live in front of faculty.
- If a faculty member types unusual input, let them; the forms validate and the audit log records it.
- Never claim an improvement the tables do not show. Say "the ablation shows no gain" when it does.
- After each session: `python scripts\reset_demo.py`.
