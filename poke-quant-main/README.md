# PokeQuant

Card info and price history come from a SQLite database (`pokequant.db`).
It starts with made-up data, and you can load REAL TCGplayer prices from inside the app.

## Run
    pip install -r requirements.txt
    streamlit run app.py
The database is built automatically on first run if it's missing.

## Loading real prices (app tab: "Load Real Prices")
Prices come from the Pokemon TCG API (pokemontcg.io), which includes TCGplayer market prices.
No key needed (a free one from dev.pokemontcg.io raises the rate limit).
1. Search a card by name (or id like base1-4), pick it, choose the print variant, and save it
   as a new card or as a replacement for an existing made-up card.
2. "Update all prices" records today's price for every tracked card. Run it daily.
3. Import past prices from any CSV with a date column and a price column.
The API only returns TODAY's price, so history builds up one day at a time unless you import a CSV.
The volatility model needs about 35 days of data; the app tells you when a card doesn't have enough yet.

## Files
- `db.py`            the only file that touches the database (auto-upgrades older database files)
- `price_sources.py` pokemontcg.io client + CSV import (all the loading logic)
- `price_ui.py`      the "Load Real Prices" screen
- `seed_cards.csv`   made-up starter cards;  `python data_generator.py` rebuilds the made-up database
- `scanner.py`, `math_model.py`, `app.py`   as before
