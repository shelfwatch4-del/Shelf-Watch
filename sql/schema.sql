-- Shelfwatch MySQL 8 schema (SRS 4.7 and 7.3). Run by scripts/load_mysql.py or manually:
--   mysql -u shelf -p shelfwatch < sql/schema.sql
-- Notes: GROUPS, RANK and KEY are reserved words in MySQL 8, so columns are named
-- feature_groups, rank_train and key_name. CHECK constraints need MySQL 8.0.16 or newer.
-- No table stores a customer identifier (only the has_customer flag).

CREATE DATABASE IF NOT EXISTS shelfwatch CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
USE shelfwatch;

-- ---------------------------------------------------------------- lineage
CREATE TABLE IF NOT EXISTS data_sources (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  name          VARCHAR(120) NOT NULL,
  role          VARCHAR(30)  NOT NULL,          -- primary, supporting, calendar
  source_url    VARCHAR(255),
  rows_raw      INT,
  date_min      DATE,
  date_max      DATE,
  license_note  VARCHAR(255)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS pipeline_log (
  step_no        INT PRIMARY KEY,
  step           VARCHAR(255) NOT NULL,
  rows_remaining INT,
  note           VARCHAR(255)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------- demand data
CREATE TABLE IF NOT EXISTS products (
  product_id        VARCHAR(20) PRIMARY KEY,
  description       VARCHAR(255),
  total_units_train INT,
  rank_train        INT
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS weekly_demand (          -- contract C1
  product_id VARCHAR(20) NOT NULL,
  week_start DATE NOT NULL,
  units      INT NOT NULL,
  orders     INT NOT NULL,
  customers  INT NOT NULL,
  avg_price  DECIMAL(10,4) NOT NULL,
  PRIMARY KEY (product_id, week_start),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS closed_weeks (week_start DATE PRIMARY KEY) ENGINE=InnoDB;   -- C1b

CREATE TABLE IF NOT EXISTS calendar_weekly (        -- C9
  week_start            DATE PRIMARY KEY,
  uk_bank_holiday_count TINYINT NOT NULL,
  trading_days_expected TINYINT NOT NULL,
  is_christmas_week     TINYINT NOT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS prior_year_weekly (      -- C10
  product_id VARCHAR(20) NOT NULL,
  week_start DATE NOT NULL,
  units      INT NOT NULL,
  orders     INT NOT NULL,
  PRIMARY KEY (product_id, week_start),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS enrichment_weekly (      -- C8
  product_id      VARCHAR(20) NOT NULL,
  week_start      DATE NOT NULL,
  gross_units     INT NOT NULL,
  cancelled_units INT NOT NULL,
  export_units    INT NOT NULL,
  guest_units     INT NOT NULL,
  PRIMARY KEY (product_id, week_start),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS transactions_top100 (    -- invoice lines of the 100 products, no customer id
  line_id         BIGINT AUTO_INCREMENT PRIMARY KEY,
  invoice_no      VARCHAR(20) NOT NULL,
  product_id      VARCHAR(20) NOT NULL,
  description     VARCHAR(255),
  quantity        INT NOT NULL,
  quantity_capped INT NOT NULL,
  invoice_date    DATETIME NOT NULL,
  week_start      DATE NOT NULL,
  unit_price      DECIMAL(10,4) NOT NULL,
  country         VARCHAR(60),
  has_customer    TINYINT(1) NOT NULL,
  is_cancelled    TINYINT(1) NOT NULL,
  INDEX idx_tx_product_week (product_id, week_start),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------- modelling results
CREATE TABLE IF NOT EXISTS model_registry (
  id               INT AUTO_INCREMENT PRIMARY KEY,
  model_name       VARCHAR(60) NOT NULL,
  target_mode      VARCHAR(30),
  feature_groups   VARCHAR(100),
  trained_on       VARCHAR(60),
  file_path        VARCHAR(255),
  library_versions JSON,
  is_active        TINYINT(1) NOT NULL DEFAULT 0,
  created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS predictions (            -- C2
  product_id    VARCHAR(20) NOT NULL,
  week_start    DATE NOT NULL,
  model_name    VARCHAR(60) NOT NULL,
  split_name    VARCHAR(10) NOT NULL,                -- test or val
  actual        DOUBLE,
  predicted     DOUBLE,
  forecast_p10  DOUBLE NULL,
  forecast_p90  DOUBLE NULL,
  PRIMARY KEY (product_id, week_start, model_name, split_name),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS metrics (                -- C3
  model_name VARCHAR(60) NOT NULL,
  split_name VARCHAR(10) NOT NULL,
  mae DOUBLE, rmse DOUBLE, wape DOUBLE, bias DOUBLE,
  PRIMARY KEY (model_name, split_name)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS tuning_trials (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  candidate     VARCHAR(80) NOT NULL,
  trial_no      INT NOT NULL,
  params        JSON,
  fold_scores   JSON,
  mean_val_wape DOUBLE,
  is_best       TINYINT(1) NOT NULL DEFAULT 0,
  INDEX idx_trials_candidate (candidate)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS ablation (
  feature_groups VARCHAR(100) NOT NULL,
  model_name     VARCHAR(60) NOT NULL,
  val_wape       DOUBLE,
  val_bias       DOUBLE,
  note           VARCHAR(255),
  PRIMARY KEY (feature_groups, model_name)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS feature_importance (
  model_name            VARCHAR(60) NOT NULL,
  feature               VARCHAR(80) NOT NULL,
  feature_group         VARCHAR(20),
  gain_importance       DOUBLE,
  mean_abs_contribution DOUBLE,
  PRIMARY KEY (model_name, feature)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS next_week_forecast (     -- C4
  product_id             VARCHAR(20) PRIMARY KEY,
  week_start             DATE NOT NULL,
  forecast_weekly        DOUBLE NOT NULL,
  forecast_p10           DOUBLE,
  forecast_p90           DOUBLE,
  sigma_error            DOUBLE,
  avg_weekly_units_train DOUBLE,
  last_avg_price         DECIMAL(10,4),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS error_analysis (
  level_name          VARCHAR(20) NOT NULL,           -- segment, week, product
  key_name            VARCHAR(80) NOT NULL,
  model_name          VARCHAR(60) NOT NULL,
  n INT, mae DOUBLE, rmse DOUBLE, wape DOUBLE, bias DOUBLE,
  over_forecast_units DOUBLE, under_forecast_units DOUBLE,
  PRIMARY KEY (level_name, key_name, model_name)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------- intelligence and backtest
CREATE TABLE IF NOT EXISTS product_intelligence (   -- C11
  product_id              VARCHAR(20) PRIMARY KEY,
  description             VARCHAR(255),
  adi DOUBLE, cv2 DOUBLE,
  pattern_class           VARCHAR(20),
  abc_class               CHAR(1),
  xyz_class               CHAR(1),
  lifecycle_stage         VARCHAR(20),
  weeks_since_last_sale   INT,
  return_rate DOUBLE, export_share DOUBLE, top3_customer_share DOUBLE,
  known_customer_coverage DOUBLE, revenue_share DOUBLE,
  FOREIGN KEY (product_id) REFERENCES products(product_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS basket_pairs (           -- C12
  product_id            VARCHAR(20) NOT NULL,
  companion_id          VARCHAR(20) NOT NULL,
  companion_description VARCHAR(255),
  n_invoices            INT NOT NULL,
  confidence            DOUBLE,
  lift                  DOUBLE,
  PRIMARY KEY (product_id, companion_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS policy_backtest (        -- C14
  policy          VARCHAR(30) NOT NULL,
  lead_time_weeks TINYINT NOT NULL,
  abc_class       VARCHAR(5) NOT NULL,                -- A, B, C or ALL
  fill_rate DOUBLE, stockout_weeks INT, avg_on_hand DOUBLE, unmet_units DOUBLE,
  PRIMARY KEY (policy, lead_time_weeks, abc_class)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS policy_backtest_weekly ( -- C15
  id              BIGINT AUTO_INCREMENT PRIMARY KEY,
  policy          VARCHAR(30) NOT NULL,
  lead_time_weeks TINYINT NOT NULL,
  product_id      VARCHAR(20) NOT NULL,
  week_start      DATE NOT NULL,
  on_hand_start DOUBLE, order_qty DOUBLE, arrived_qty DOUBLE,
  demand DOUBLE, sold DOUBLE, unmet DOUBLE, on_hand_end DOUBLE,
  INDEX idx_pbw_product_policy (product_id, policy)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------- system tables (written by the API)
CREATE TABLE IF NOT EXISTS inventory_levels (
  product_id      VARCHAR(20) PRIMARY KEY,
  on_hand         INT NOT NULL,
  lead_time_weeks TINYINT NOT NULL DEFAULT 1,
  service_level   DECIMAL(4,3) NOT NULL DEFAULT 0.950,
  updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT chk_on_hand CHECK (on_hand >= 0),
  CONSTRAINT chk_lead    CHECK (lead_time_weeks BETWEEN 1 AND 6),
  CONSTRAINT chk_service CHECK (service_level BETWEEN 0.80 AND 0.99),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS replenishment_plans (
  id          INT AUTO_INCREMENT PRIMARY KEY,
  week_start  DATE NOT NULL,
  status      ENUM('Draft','Approved','Cancelled') NOT NULL DEFAULT 'Draft',
  note        VARCHAR(255),
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
  approved_at DATETIME NULL
) ENGINE=InnoDB;

-- Proposed order lines of a plan. Rows become purchase_orders only when the plan is approved.
CREATE TABLE IF NOT EXISTS plan_lines (
  id         INT AUTO_INCREMENT PRIMARY KEY,
  plan_id    INT NOT NULL,
  product_id VARCHAR(20) NOT NULL,
  quantity   INT NOT NULL,
  CONSTRAINT chk_line_qty CHECK (quantity > 0),
  INDEX idx_pl_plan (plan_id),
  FOREIGN KEY (plan_id) REFERENCES replenishment_plans(id),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS purchase_orders (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  plan_id       INT NULL,
  product_id    VARCHAR(20) NOT NULL,
  quantity      INT NOT NULL,
  supplier_note VARCHAR(255),
  status        ENUM('Pending','Approved','Received','Cancelled') NOT NULL DEFAULT 'Pending',
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT chk_qty CHECK (quantity > 0),
  INDEX idx_po_status (status),
  INDEX idx_po_product (product_id),
  FOREIGN KEY (plan_id) REFERENCES replenishment_plans(id),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS prediction_log (
  id          BIGINT AUTO_INCREMENT PRIMARY KEY,
  request_id  VARCHAR(40),
  endpoint    VARCHAR(40),
  product_id  VARCHAR(20) NULL,
  inputs      JSON,
  outputs     JSON,
  model_name  VARCHAR(60),
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_pl_created (created_at)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS api_logs (
  id          BIGINT AUTO_INCREMENT PRIMARY KEY,
  request_id  VARCHAR(40),
  method      VARCHAR(10),
  path        VARCHAR(255),
  status      SMALLINT,
  duration_ms DECIMAL(10,2),
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_log_created (created_at)
) ENGINE=InnoDB;
