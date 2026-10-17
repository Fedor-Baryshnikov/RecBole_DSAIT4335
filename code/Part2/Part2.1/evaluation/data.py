"""Load Task 2.1 inputs exported by the shared RecBole repository."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _plain_field(field: str) -> str:
    return field.split(":", 1)[0]


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        reader.fieldnames = [_plain_field(field) for field in reader.fieldnames]
        return [dict(row) for row in reader]


def load_interactions(path: str | Path) -> list[dict[str, str]]:
    rows = read_tsv(path)
    required = {"user_id", "item_id"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain user_id and item_id")
    return rows


def ground_truth_by_user(
    rows: Iterable[dict[str, str]],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        result[row["user_id"]].add(row["item_id"])
    return dict(result)


def history_by_user(rows: Iterable[dict[str, str]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        result[row["user_id"]].append(row["item_id"])
    return dict(result)


def load_recommendations(
    path: str | Path, k: int
) -> tuple[dict[str, object], dict[str, list[str]]]:
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("recommendations"), dict
    ):
        raise ValueError(f"{path} is not a repository recommendation JSON")
    stored_k = payload.get("k")
    if stored_k is not None and int(stored_k) < k:
        raise ValueError(f"{path} contains top-{stored_k}, not top-{k}")
    if payload.get("split") not in (None, "test"):
        raise ValueError(f"{path} contains {payload['split']} recommendations, not test")

    recommendations: dict[str, list[str]] = {}
    for user_id, row in payload["recommendations"].items():
        if not isinstance(row, dict) or not isinstance(row.get("items"), list):
            raise ValueError(f"invalid recommendation row for user {user_id}")
        items = [str(item) for item in row["items"][:k]]
        if len(items) != k:
            raise ValueError(f"user {user_id} has fewer than {k} recommendations")
        if len(items) != len(set(items)):
            raise ValueError(f"user {user_id} has duplicate recommendations")
        recommendations[str(user_id)] = items
    metadata = {key: value for key, value in payload.items() if key != "recommendations"}
    return metadata, recommendations


def load_item_genres(path: str | Path) -> dict[str, frozenset[str]]:
    rows = read_tsv(path)
    if rows and "item_id" not in rows[0]:
        raise ValueError(f"{path} must contain item_id")
    if rows and "class" not in rows[0] and "genres" not in rows[0]:
        raise ValueError(f"{path} must contain class or genres")
    result = {}
    for row in rows:
        raw = row.get("class", row.get("genres", "")).replace("|", " ")
        result[row["item_id"]] = frozenset(part for part in raw.split() if part)
    return result


def age_group(raw_age: str) -> str:
    age = int(raw_age)
    if age < 18:
        return "under_18"
    if age <= 24:
        return "18_24"
    if age <= 34:
        return "25_34"
    if age <= 44:
        return "35_44"
    if age <= 49:
        return "45_49"
    if age <= 55:
        return "50_55"
    return "56_plus"


def load_user_groups(path: str | Path) -> dict[str, dict[str, str]]:
    rows = read_tsv(path)
    if rows and "user_id" not in rows[0]:
        raise ValueError(f"{path} must contain user_id")
    result = {}
    for row in rows:
        attributes = {
            key: value for key, value in row.items() if key != "user_id"
        }
        if attributes.get("age") not in (None, ""):
            attributes["age"] = age_group(attributes["age"])
        result[row["user_id"]] = attributes
    return result
