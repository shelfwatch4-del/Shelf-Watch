# STAKEHOLDER.md: Retail Inventory and Replenishment Manager

Save as `docs/STAKEHOLDER.md`. Use it for the report, slides, demo and viva. SRS reference: section 1.6.

## 1. Decision record
| Item | Decision |
|---|---|
| Primary and only designed-for stakeholder | **Retail Inventory and Replenishment Manager** |
| Why | The problem statement asks which products may need inventory attention; that is this person's weekly question. Every output (forecast, range, risk, order quantity, reorder date) maps to a decision they own. One user keeps the website and report coherent |
| Chosen by | The team, after the mentor asked for one stakeholder |
| Not designed for | Finance, merchandising, warehouse staff (no separate screens, roles or reports; they benefit indirectly) |
| Status | The persona is **assumed** from the problem statement and standard replenishment practice. It is not from interviews. Say so in the report |

## 2. Persona card
| Attribute | Description |
|---|---|
| Role | Manager at a UK online giftware wholesaler and retailer (the business behind the dataset) |
| Owns | The weekly replenishment plan for about 100 fast-moving products, purchase-order approval, stock-cover targets |
| Decisions | What to order, how much, when; which products to stop reordering; how much safety stock to hold; how bad a supplier delay would be |
| Rhythm | Weekly review before supplier orders; daily exception checks |
| Pain points | Best sellers running out (lost sales); slow or discontinued items piling up (tied-up cash); bulk wholesale orders distorting averages; supplier delays; no single view of forecast, stock and open orders |
| Success measures | Fewer stockout weeks, higher fill rate, days of cover within target, less excess stock, a forecast error they can trust |
| Constraints | Supplier lead time 1 to 4 weeks; no reliable on-hand feed, so stock is entered or simulated |

## 3. Their weekly routine and the page that serves each step
| Step | Question | Page |
|---|---|---|
| 1 See what needs attention | Which products will run short before the next delivery? | Home: headline, shelf wall, Action list |
| 2 Understand one product | How much will it sell, and how uncertain is that? | Predict |
| 3 Decide the order | How much, and by when? | Home and Predict (recommended quantity, days to reorder) |
| 4 Act | Record stock, raise purchase orders | Inventory manager |
| 5 Test assumptions | What if the supplier is a week late? | Predict: what-if simulator |
| 6 Set policy | Which items need closer control or should stop being reordered? | Product intelligence |
| 7 Trust the numbers | How accurate is the forecast? | Model lab |

## 4. Features reframed for this person
| Feature | Why the manager cares |
|---|---|
| Prediction interval (low to high) | Orders are placed against uncertainty, not a single number |
| Revenue at risk | Ranks which shortfall to fix first |
| Action list (Order now, Order soon, Review to stop, Hold) | Turns analysis into a to-do list |
| ABC and XYZ class | Sets review frequency: A items need weekly attention |
| Lifecycle stage | Stops ordering for inactive or declining items |
| Return rate | Net demand is lower than gross sales for returned items |
| Bought-together links | Companion products to check when one is at risk |
| Prior-year and holiday features | Seasonal build-up and short trading weeks change what to order |
| User-entered stock and purchase orders (CRUD) | Replaces the simulated stock with what is really on the shelf |

## 5. Evidence that the system helps (to quote in the report)
Use only numbers produced by the project:
1. **Forecast trust:** WAPE and bias on the 8 test weeks, and interval coverage (target about 80 percent).
2. **Policy backtest (FR-R-9):** fill rate, stockout weeks and average on-hand under four policies (naive, moving average, best model, best model with safety stock). This is the manager's own success measure. Report it even if differences are small.
3. **Risk and revenue at risk** at the default settings, with the sensitivity table.
4. **Honest limits:** stock is simulated unless entered; one year of main data; intermittent items; the persona is assumed.

## 6. Validate the assumption (30 minutes, optional but valuable)
Ask the mentor, and if possible one person who works in retail or stock control:
1. What do you check first each week when planning orders?
2. What do you do when a best seller is about to run out?
3. How do you decide how much safety stock to hold?
4. What would make you trust or distrust a forecast?
5. Which of these screens would you actually use, and what is missing?
Write the answers in the report under "Persona validation". If you cannot interview anyone, state that the persona is assumed.

## 7. Two-sentence pitch
"Shelfwatch gives a retail inventory and replenishment manager a weekly forecast with an honest range, tells them which products will run short, how much to order and by when, and lets them record stock and purchase orders in one place. Everything is built around their weekly routine, and its value is measured with a replenishment backtest, not only forecast error."
