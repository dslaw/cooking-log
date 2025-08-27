# cooking-log

Structure and analyze cooking log data.

The cooking log should be present as a markdown formatted file located at
`data/cooking-log.md`. The cooking log has the following structure:

```markdown
# May 31, 2025

## Lunch

Pizza marinara.

Roasted kobucha squash and broccoli with peanut butter sauce.

## Dinner

Tacos de frijoles refritos.

Kobucha squash all'aglio e olio.

Notes:
- For tacos, smeared refried beans on tortillas then topped with cabbage,
  pickled carrot and onion, and cilantro.
```

Each day in the log has a header with the date, then a sub-header indicating
the meal. Only "Lunch" and "Dinner" are supported for meal sub-headers. Each
item within an entry (unique over date and meal) represents a single dish that
was cooked. An entry may also have an optional "Notes" section.


# Getting Started

To get started, first create a virtual environment using `uv`:
```bash
$ uv sync
```

Then, set up a directory for data artifacts and place the cooking log file in
it:
```bash
$ mkdir -p data
$ cp path/to/cooking-log.md data/
```

Finally, run the Python scripts:
```bash
# Remove any data artifacts that are present.
$ rm data/*.parquet
$ uv run python -m src.parse_entries
$ uv run python -m src.calculate_distances
$ uv run python -m src.find_cliques
$ uv run python -m src.format_cliques
```
