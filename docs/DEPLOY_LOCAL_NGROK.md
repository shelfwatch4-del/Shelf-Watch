# DEPLOY_LOCAL_NGROK.md: Local server, MySQL, public URL and Postman

Follows the mentor's four steps: (1) train and save the model, (2) serve it locally with FastAPI, (3) create a public URL, (4) test with Postman. Our project adds MySQL for all data. SRS references: 11.1 to 11.8.

## 0. What runs where
| Part | Runs on | Address |
|---|---|---|
| MySQL 8 | The demo laptop | localhost:3306 |
| FastAPI + the five web pages | The demo laptop | http://localhost:8000 |
| ngrok tunnel | The demo laptop | https://YOUR-DOMAIN.ngrok-free.app (public) |
| Postman | Any computer | Uses the local or the public address |

The public URL works only while the laptop, MySQL, FastAPI and ngrok are all running.

## 1. Install (Windows)
1. Python 3.11 (tick "Add Python to PATH"), Git, Postman.
2. **MySQL Community Server 8.x** and MySQL Workbench from mysql.com. Remember the root password. (XAMPP installs MariaDB, not MySQL; use MySQL 8 because the mentor asked for MySQL and the schema uses MySQL 8 features.)
3. An ngrok account and the ngrok agent from ngrok.com. Run once: `ngrok config add-authtoken <your token from the ngrok dashboard>`.

## 2. Create the database and user
Open MySQL Workbench (or `mysql -u root -p`) and run:
```sql
CREATE DATABASE shelfwatch CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
CREATE USER 'shelf'@'localhost' IDENTIFIED BY 'change_me';
GRANT ALL PRIVILEGES ON shelfwatch.* TO 'shelf'@'localhost';
FLUSH PRIVILEGES;
```
Check: `mysql -u shelf -p shelfwatch` then `SHOW TABLES;` (empty at first).

## 3. Configure the project
```
copy .env.example .env
```
Edit `.env`: set `DATABASE_URL=mysql+pymysql://shelf:change_me@localhost:3306/shelfwatch` and your own `API_KEY`. URL-encode special characters in the password (`@` becomes `%40`). `.env` is never committed.
Install: `venv\Scripts\pip install -r requirements.txt` (includes `pymysql`, `cryptography`, `python-dotenv`; MySQL 8's default login method needs `cryptography`).

## 4. Build everything and load MySQL
```
venv\Scripts\python -m src.data_prep --raw "data\raw\Online Retail.xlsx"
venv\Scripts\python -m src.enrichment
venv\Scripts\python -m src.intelligence
venv\Scripts\python -m src.train
venv\Scripts\python -m src.backtest
venv\Scripts\python scripts\load_mysql.py
```
`load_mysql.py` creates the tables from `sql/schema.sql` and loads every output. It prints the row count per table. Open Workbench and look at the tables: this is part of the evaluation demo.

## 5. Run the server locally (mentor step 2)
```
scripts\start_server.bat
```
(or `venv\Scripts\python -m uvicorn api.main:app --reload --port 8000`)
Open and check:
- http://localhost:8000/ : the website (Home, no login)
- http://localhost:8000/docs : Swagger documentation
- http://localhost:8000/health : `{"status":"ok", ...}`

## 6. Test with Postman locally (mentor step 4, first pass)
1. Postman: Import `postman/Shelfwatch.postman_collection.json` and `postman/Shelfwatch.local.postman_environment.json`. Select the environment "Shelfwatch local" (top right). Set `apiKey` to the value in your `.env`.
2. Folder **00 Mentor flow**: `GET {{baseUrl}}/health` then `POST {{baseUrl}}/predict` with the JSON body `{"product_id": "{{productId}}", "lead_time_weeks": 1, "service_level": 0.95}`. You should see 200 and a forecast with `forecast_p10`, `forecast_p90`, `risk_level` and `action`.
3. Run the whole collection: Collection, **Run**, Run Shelfwatch API tests. Every test should pass. Screenshot the runner result.

## 7. Create the public URL (mentor step 3)
In a second terminal (server still running):
```
scripts\start_ngrok.bat
```
(or `ngrok http 8000`). ngrok prints a **Forwarding** address like `https://xxxx.ngrok-free.app`. If your ngrok dashboard shows an assigned dev domain, use the command it displays for that domain (flag names differ between versions).

Open `https://<your-address>/docs` and `/health` in a browser. On the free plan a one-time warning page appears: click **Visit Site**. Programs and Postman avoid it by sending the header `ngrok-skip-browser-warning: true`; our collection and our pages already send it.

## 8. Test the public URL with Postman (mentor step 4, second pass)
1. Import `postman/Shelfwatch.public.postman_environment.json`, set `baseUrl` to your ngrok address (no trailing slash) and `apiKey`.
2. Select "Shelfwatch public (ngrok)" and run folder 00 and then the whole collection. Screenshot the results next to the local run.
3. Open the public Home page on a phone to confirm it works from outside your laptop.

## 9. Backup and restore
```
mysqldump -u shelf -p shelfwatch > backup_YYYYMMDD.sql
mysql -u shelf -p shelfwatch < backup_YYYYMMDD.sql
```
Keep the dump and the project folder on a second laptop. `python scripts\reset_demo.py` resets stock, plans, orders and logs to the seeded demo state.

## 10. Troubleshooting
| Symptom | Likely cause and fix |
|---|---|
| `Access denied for user 'shelf'` | Wrong password in `.env`, or the user was not granted on `shelfwatch.*`; redo section 2 |
| `Can't connect to MySQL server` | MySQL service not running: Windows Services, MySQL80, Start |
| `cryptography package is required` | `pip install cryptography` |
| Password with `@` or `#` fails | URL-encode it in `DATABASE_URL` |
| `Address already in use` on 8000 | `netstat -ano | findstr :8000` then end that process, or use another port and `ngrok http <port>` |
| ngrok shows an error page | The local server is not running on that port, or the authtoken is missing |
| Browser shows the ngrok warning page | Click Visit Site once; Postman sends the skip header |
| Pages load but data is empty | `scripts\load_mysql.py` was not run, or `.env` points to a different database |
| 401 on writes in Postman | `apiKey` in the environment differs from `API_KEY` in `.env` |
| Public URL stops during a demo | Laptop slept or lost network; keep power on, sleep off, and use the local address as a fallback |
