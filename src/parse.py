import datetime
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Meal(StrEnum):
    Lunch = "Lunch"
    Dinner = "Dinner"


@dataclass(frozen=True)
class CookingLogEntry:
    date: datetime.date
    meal: Meal
    dishes: list[str]
    notes: list[str]


def read_cooking_log(input_file: Path) -> list[str]:
    return input_file.read_text().split("\n")


def parse_date(line: str) -> datetime.date:
    return datetime.datetime.strptime(line, "# %B %d, %Y").date()


def trim_entry_body(lines: list[str]) -> list[str]:
    start_idx = 0
    end_idx = len(lines) - 1

    while start_idx < len(lines) and not lines[start_idx].strip():
        start_idx += 1

    while end_idx >= 0 and not lines[end_idx].strip():
        end_idx -= 1

    if start_idx <= end_idx:
        return lines[start_idx : end_idx + 1]
    return []


def split_entries(lines: list[str]) -> list[tuple[datetime.date, Meal, list[str]]]:
    entries: list[tuple[datetime.date, Meal, list[str]]] = []
    current_date: datetime.date | None = None
    current_meal: Meal | None = None
    current_body: list[str] = []

    def add_entry(
        date: datetime.date | None, meal: Meal | None, body: list[str]
    ) -> None:
        if date is None or meal is None:
            return

        trimmed_body = trim_entry_body(body)
        if trimmed_body:
            entries.append((date, meal, trimmed_body))

        return

    for line in lines:
        if line.startswith("# "):
            # New date found, save previous entry if it has content.
            add_entry(current_date, current_meal, current_body)

            current_date = parse_date(line)
            current_meal = None
            current_body = []

        elif line.startswith("## "):
            # New meal found, save previous entry if it has content.
            add_entry(current_date, current_meal, current_body)

            current_body = []

            meal_name = line[3:].strip()

            try:
                current_meal = Meal(meal_name)
            except ValueError:
                continue

        else:
            # Part of entry body.
            if current_date is not None and current_meal is not None:
                current_body.append(line)

    # Don't forget the last entry.
    add_entry(current_date, current_meal, current_body)

    return entries


def parse_body(body: list[str]) -> tuple[list[str], list[str]]:
    it = iter(body)

    dishes: list[str] = []
    current_dish: str | None = None
    has_notes = False
    for line in it:
        if line.startswith("Notes"):
            has_notes = True
            break

        if current_dish is None and line:
            current_dish = line
        elif current_dish is not None and line:
            current_dish = f"{current_dish} {line}"
        elif current_dish is not None and not line:
            dishes.append(current_dish)
            current_dish = None
        else:  # current_dish is None and not line
            pass

    if current_dish is not None:
        dishes.append(current_dish)

    notes: list[str] = list(filter(None, it)) if has_notes else []
    return dishes, notes


def parse_cooking_log(lines: list[str]) -> list[CookingLogEntry]:
    entries: list[CookingLogEntry] = []
    for date, meal, raw_body in split_entries(lines):
        dishes, notes = parse_body(raw_body)
        entries.append(
            CookingLogEntry(
                date=date,
                meal=meal,
                dishes=dishes,
                notes=notes,
            )
        )

    return entries
