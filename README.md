# CourtIQ

CourtIQ projects points, rebounds, assists and their sum (PRA) for a player's next NBA regular-season appearance. It covers Luka Dončić, Kyrie Irving, Jayson Tatum, Shai Gilgeous-Alexander and Jalen Brunson.

## Run the dashboard

Use Python 3.10 or 3.11. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The historical demo works offline from the existing `database_24_25.csv`. Its 212 appearances for these five players span part of 2024–25 and end on February 7, 2025. Select a player, choose a hypothetical date and select **Expected to play** to see a projection. The Luka demo uses a five-game average because the bundled data has too few of his appearances for Ridge. Its baseline comparison uses four validation appearances.

## Collect and train on NBA game logs

```bash
python -m src.fetch_data
python -m src.train
python -m streamlit run app.py
```

The collector requests NBA Stats [PlayerGameLog](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/playergamelog.md) for 2022–23 through 2026–27. It skips seasons before October 1 of their start year and uses only returned regular-season appearances. An ongoing season is partial by definition. The endpoint returns game logs rather than a schedule or injury feed.

Per-player and season responses are cached in `data/raw/nba/`. Current caches are reused for 12 hours. Historical caches fetched after July 1 following the season are reused indefinitely. The collector uses sequential requests, timeouts and bounded retries with backoff. For HTTP 429 it respects a numeric Retry-After within its wait budget. It stops requesting after three consecutive player-season failures. It publishes `data/games.csv` only after coverage checks pass. Empty, schema-valid logs represent zero recorded appearances. An empty or truncated response cannot replace a cache containing additional game IDs.

Use `python -m src.fetch_data --refresh` to refresh all eligible caches. If NBA Stats is unavailable, use `python -m src.fetch_data --offline` to rebuild from existing caches, including stale ones. Offline collection reports stale caches and requires a file for every eligible player-season, including any seasons with zero appearances. The existing snapshot is preserved on collection failure. Run the collector from a network where NBA Stats is reachable, then copy the cache or normalized snapshot to the machine running CourtIQ if needed. The original project CSV and any older `data/raw/luka_2022_23.csv` can coexist with the new cache.

Training writes `artifacts/courtiq.json` with coefficients, feature order, holdout scores and a fingerprint of the input. The dashboard and CLI reject an artifact if the snapshot changes. Run training after every successful refresh. Keep the snapshot and dependency versions with any reported results. For an offline demo run:

```bash
python -m src.train --demo
python -m src.predict --demo --player 1628369 --date 2025-02-08 --availability expected
```

For an NBA projection, use `python -m src.predict --player ID --date YYYY-MM-DD --availability expected` with the actual upcoming game date. IDs are listed by `python -m src.predict --help` and in `src/data.py`. Check the game date and injury status independently. Availability starts unconfirmed. CourtIQ withholds projections for out or unconfirmed players, players without sufficient history and NBA projections more than 30 days after the last recorded appearance. Missed games never become zero-stat training rows. Days since appearance can include rest, injury, schedule gaps or the offseason.

## How the model works

1. `src/fetch_data.py` collects logs and `src/data.py` validates them, keeps positive-minute appearances and rejects conflicting duplicates.
2. `src/model.py` builds features player by player. For a target appearance it shifts points, rebounds, assists, minutes, field-goal attempts and free-throw attempts by one appearance before computing rolling averages over 3, 5 and 10 appearances. Days since the previous appearance is capped at 30. Inference appends a date-only row and runs that same feature code.
3. `src/train.py` uses the earliest 80% of eligible rows per player to fit a standardized Ridge model with fixed alpha 20. The latest 20% (rounded up) is the holdout. Its predictions use the scores of earlier appearances, including earlier holdout appearances, as those results would become available in real use. Model coefficients and scaling stay fixed throughout the holdout. It requires at least 20 training and 5 holdout rows, then refits on all eligible history for upcoming projections.
4. The validation table compares Ridge, the last appearance and the last five appearances' mean on identical holdout games. Lower mean absolute error (MAE) is better. PRA MAE measures the error of the summed estimates directly. The error table describes that holdout and can vary by player, target and sample period.

### Example measured result

On the historical demo, Jalen Brunson's eight holdout appearances produced PTS MAE of 7.27 for Ridge, 12.50 for last appearance and 8.22 for the five-game average. His REB MAE was 2.05, 2.25 and 1.70, respectively. Run the demo training command for all measured results and sample sizes.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests run without NBA Stats. They cover feature leakage, chronological holdout, reference baseline and PRA calculations, availability, stale-history safeguards, response decoding, retry and cache behavior, and Streamlit projections. GitHub Actions runs the tests and demo commands on pushes and pull requests.

The original notebook remains in the repository. Its plotting cells also require `matplotlib`, which can be installed separately when working with that notebook.
