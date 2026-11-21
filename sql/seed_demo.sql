-- Seed demo data for Shelfwatch system tables (SRS 7.3, FR-DB-11)
-- 5 stock records and 4 purchase orders across all 4 statuses.

USE shelfwatch;

-- Ensure demo products exist
INSERT INTO products (product_id, description, total_units_train, rank_train)
VALUES
  ('85123A', 'WHITE HANGING HEART T-LIGHT HOLDER', 1000, 1),
  ('22423', 'REGENCY CAKESTAND 3 TIER', 800, 2),
  ('85099B', 'JUMBO BAG RED RETROSPOT', 600, 3),
  ('84879', 'ASSORTED COLOUR BIRD ORNAMENT', 500, 4),
  ('47566', 'PARTY BUNTING', 400, 5)
ON DUPLICATE KEY UPDATE
  description = VALUES(description);

-- 5 inventory level records
INSERT INTO inventory_levels (product_id, on_hand, lead_time_weeks, service_level)
VALUES
  ('85123A', 150, 1, 0.950),
  ('22423', 80, 2, 0.950),
  ('85099B', 0, 1, 0.990),
  ('84879', 220, 3, 0.900),
  ('47566', 45, 1, 0.950)
ON DUPLICATE KEY UPDATE
  on_hand = VALUES(on_hand),
  lead_time_weeks = VALUES(lead_time_weeks),
  service_level = VALUES(service_level);

-- 4 purchase orders covering all four lifecycle statuses
INSERT INTO purchase_orders (product_id, quantity, supplier_note, status)
VALUES
  ('85123A', 100, 'Weekly restock', 'Pending'),
  ('22423', 50, 'Approved replenishment', 'Approved'),
  ('85099B', 200, 'Received order', 'Received'),
  ('84879', 75, 'Cancelled duplicate order', 'Cancelled');
