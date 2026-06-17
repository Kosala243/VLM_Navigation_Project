"""goal_parser.py
Flexible goal parser for generalized indoor room/location navigation.
It does not force every goal into building/floor/room.

Updated goal parsing approach:
- Keep the raw goal as the primary target evidence.
- Extract searchable tokens and aliases for different room-code styles.
- Generate hypotheses, not fixed conclusions.
- Use lightweight rules first, then optionally let the VLM refine the same schema.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .model_loader import ModelWrapper


@dataclass
class GoalHypothesis:
    interpretation: str
    evidence_needed: list[str] = field(default_factory=list)
    confidence: str = "medium"  # high | medium | low


@dataclass
class NavigationGoal:
    raw_goal: str
    target_type: str = "room_or_location"
    known_tokens: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    possible_interpretations: list[GoalHypothesis] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    search_strategy: str = "Use signs, directories, room labels, front-desk/staff help if available, and frontier exploration."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def compact(self) -> str:
        hyps = "; ".join(h.interpretation for h in self.possible_interpretations[:4]) or "unknown"
        return (
            f"Goal={self.raw_goal}; tokens={self.known_tokens}; aliases={self.aliases}; "
            f"hypotheses={hyps}; constraints={self.constraints}"
        )


class GoalParser:
    _PROMPT = """\
You are parsing a robot indoor navigation goal for an unknown office/university/hospital/airport/public building.
Do NOT assume all buildings use the same room-code format.
Your job is NOT to decide one final meaning. Your job is to preserve searchable labels and produce possible interpretations.

Important principles:
- Keep the exact raw target as the strongest search label.
- Generate aliases for likely label variants, e.g. "3.14" <-> "314", "F-204" <-> "F204", "C2.005" <-> "C2-005".
- Treat building/floor/room fields as possible constraints only, not facts.
- For named places such as departments, labs, corridors, gates, suites, pharmacies, or reception, preserve the name as target_name or possible_zone.
- Evidence should mention signs, directories, room plates, floor indicators, corridor/zone signs, or official front-desk/staff help.

Return ONLY valid JSON with this exact schema:
{
  "target_type": "room_or_location | person_office | department | facility | suite_or_unit | gate_or_zone | lab | unknown",
  "known_tokens": ["important raw tokens to search for on signs/doors"],
  "aliases": ["alternative labels that may appear on signs/doors"],
  "constraints": {
    "possible_building": null,
    "possible_floor": null,
    "possible_room": null,
    "possible_zone": null,
    "target_name": null
  },
  "possible_interpretations": [
    {
      "interpretation": "short explanation",
      "evidence_needed": ["what evidence confirms or rejects this"],
      "confidence": "high | medium | low"
    }
  ],
  "search_strategy": "one sentence strategy for navigation"
}

Goal: {goal}
"""

    def __init__(self, model: "ModelWrapper | None" = None):
        self.model = model

    def parse(self, goal: str) -> NavigationGoal:
        """Parse a navigation goal into flexible searchable tokens, aliases, and hypotheses.

        Rules give a deterministic safe baseline. If a model is provided, the VLM may refine
        the same schema. Any missing or malformed VLM fields fall back to the rule output.
        """
        goal = goal.strip()
        rule_goal = self._parse_with_rules(goal)

        if self.model is None:
            return rule_goal

        prompt = self._PROMPT.replace("{goal}", goal)
        response = self.model.query(prompt, max_new_tokens=700)
        data = _extract_json(response)
        if not data:
            return rule_goal

        try:
            hyps = []
            for h in data.get("possible_interpretations", []) or []:
                if not isinstance(h, dict):
                    continue
                hyps.append(GoalHypothesis(
                    interpretation=str(h.get("interpretation", "")).strip(),
                    evidence_needed=_unique_strings(h.get("evidence_needed", [])),
                    confidence=_normalize_confidence(h.get("confidence", "medium")),
                ))
            hyps = [h for h in hyps if h.interpretation]

            constraints = _merge_constraints(
                rule_goal.constraints,
                data.get("constraints", {}) if isinstance(data.get("constraints", {}), dict) else {},
            )

            known_tokens = _unique_strings(
                [goal]
                + data.get("known_tokens", [])
                + rule_goal.known_tokens
            )
            aliases = _unique_strings(
                data.get("aliases", [])
                + rule_goal.aliases
            )

            return NavigationGoal(
                raw_goal=goal,
                target_type=str(data.get("target_type", rule_goal.target_type)).strip() or rule_goal.target_type,
                known_tokens=known_tokens,
                aliases=aliases,
                constraints=constraints,
                possible_interpretations=hyps or rule_goal.possible_interpretations,
                search_strategy=str(data.get("search_strategy", rule_goal.search_strategy)).strip()
                or rule_goal.search_strategy,
            )
        except Exception:
            return rule_goal

    def _parse_with_rules(self, goal: str) -> NavigationGoal:
        text = goal.strip()
        lower = text.lower()

        constraints: dict[str, Any] = _empty_constraints()
        hyps: list[GoalHypothesis] = []
        aliases: list[str] = []
        target_type = "room_or_location"

        # Strong baseline: always keep exact target and searchable chunks.
        tokens = _extract_tokens(text)

        # 1) University/building-style code: C2.005, B3_210, A1-101.
        # Keep as hypothesis, not fact.
        m = re.search(r"\b([A-Za-z]+)(\d+)[.\-_](\d+[A-Za-z]*)\b", text)
        if m:
            b, f, r = m.group(1).upper(), m.group(2), m.group(3).upper()
            constraints.update({"possible_building": b, "possible_floor": f, "possible_room": r})
            compact = f"{b}{f}{r}"
            aliases.extend([text, f"{b}{f}.{r}", f"{b}{f}-{r}", compact, f"Room {compact}"])
            hyps.append(GoalHypothesis(
                interpretation=f"May mean Building/Zone {b}, Floor {f}, Room {r}.",
                evidence_needed=[f"building or zone sign for {b}", f"floor indicator {f}", f"door label matching {text} or {compact}"],
                confidence="medium",
            ))
            hyps.append(GoalHypothesis(
                interpretation=f"May simply be the full room label/code '{text}' without separable building/floor meaning.",
                evidence_needed=[f"door label or directory entry exactly matching {text}"],
                confidence="medium",
            ))

        # 3) Floor.room format: 3.14, 2.005.
        # This must be checked before generic number-only extraction.
        m = re.search(r"\b(\d+)[.](\d+[A-Za-z]*)\b", text)
        if m and not re.search(r"[A-Za-z]+\d+[.]\d+", text):
            floor, room = m.group(1), m.group(2).upper()
            constraints["possible_floor"] = constraints.get("possible_floor") or floor
            constraints["possible_room"] = constraints.get("possible_room") or room
            aliases.extend([f"{floor}.{room}", f"{floor}{room}", f"Room {floor}{room}", f"Room {floor}.{room}"])
            hyps.append(GoalHypothesis(
                interpretation=f"May mean Floor {floor}, Room {room}, or a full room label '{floor}.{room}'.",
                evidence_needed=[f"floor indicator {floor}", f"door label {floor}.{room} or {floor}{room}", "directory listing that confirms the notation"],
                confidence="medium",
            ))

        # 4) Hyphenated zone/building code: F-204, B-12.
        m = re.search(r"\b([A-Za-z]+)[-](\d+[A-Za-z]*)\b", text)
        if m and not re.search(r"[A-Za-z]+\d+[.\-_]\d+", text):
            zone, room = m.group(1).upper(), m.group(2).upper()
            constraints["possible_zone"] = constraints.get("possible_zone") or zone
            constraints["possible_room"] = constraints.get("possible_room") or room
            aliases.extend([f"{zone}-{room}", f"{zone}{room}", f"Room {zone}-{room}", f"Room {zone}{room}"])
            hyps.append(GoalHypothesis(
                interpretation=f"May be a full label '{zone}-{room}', or Zone {zone} with room {room}.",
                evidence_needed=[f"zone/sign for {zone}", f"door label {zone}-{room} or {zone}{room}"],
                confidence="medium",
            ))

        # 5) Generic room number with explicit word: Room 204.
        m = re.search(r"\broom\s+([A-Za-z]*\d+[A-Za-z]*)\b", text, re.I)
        if m:
            room = m.group(1).upper()
            constraints["possible_room"] = constraints.get("possible_room") or room
            aliases.extend([room, f"Room {room}"])
            hyps.append(GoalHypothesis(
                interpretation=f"Target is likely a room label containing '{room}'.",
                evidence_needed=[f"door label {room}", f"corridor range containing {room}", "directory listing"],
                confidence="medium",
            ))

        # 6) Suite/unit: Suite 4B, Unit E36, apartment-like E36.
        m = re.search(r"\b(suite|unit|apartment|apt)\s+([A-Za-z]*\d+[A-Za-z]*)\b", text, re.I)
        if m:
            target_type = "suite_or_unit"
            label = m.group(2).upper()
            constraints["possible_room"] = constraints.get("possible_room") or label
            aliases.extend([label, f"{m.group(1).title()} {label}", f"Suite {label}", f"Unit {label}"])
            hyps.append(GoalHypothesis(
                interpretation=f"Target is likely a suite/unit label '{label}'.",
                evidence_needed=[f"suite/unit label {label}", "directory listing", "corridor range sign"],
                confidence="medium",
            ))

        # 7) Gate style: Gate B12.
        m = re.search(r"\bgate\s+([A-Za-z]*\d+[A-Za-z]*)\b", text, re.I)
        if m:
            target_type = "gate_or_zone"
            gate = m.group(1).upper()
            constraints["possible_zone"] = constraints.get("possible_zone") or gate
            aliases.extend([gate, f"Gate {gate}"])
            hyps.append(GoalHypothesis(
                interpretation=f"Target is likely gate/zone '{gate}'.",
                evidence_needed=[f"gate sign {gate}", "directional signs for gates", "directory/map"],
                confidence="medium",
            ))

        # 8) Lab or named zone: Lab 2-West.
        m = re.search(r"\blab\s+([A-Za-z0-9]+(?:[- ]+[A-Za-z0-9]+)*)\b", text, re.I)
        if m:
            target_type = "lab"
            zone = m.group(1).strip()
            constraints["possible_zone"] = constraints.get("possible_zone") or zone
            constraints["target_name"] = constraints.get("target_name") or text
            aliases.extend([text, f"Lab {zone}"])
            hyps.append(GoalHypothesis(
                interpretation=f"Target is a named lab/zone: '{text}'.",
                evidence_needed=[f"lab sign for {zone}", "department/lab directory", "corridor or zone sign"],
                confidence="medium",
            ))

        # 9) Hospital/dept/corridor/level style: Apotheek MUMC+ niveau 1 corridor 8.
        if re.search(r"\b(niveau|level|floor)\s*\d+\b", lower) or re.search(r"\bcorridor\s*\d+\b", lower):
            target_type = "department" if not re.search(r"\broom\b", lower) else target_type
            level_match = re.search(r"\b(?:niveau|level|floor)\s*(\d+)\b", lower, re.I)
            corridor_match = re.search(r"\bcorridor\s*(\d+[A-Za-z]*)\b", lower, re.I)
            if level_match:
                constraints["possible_floor"] = constraints.get("possible_floor") or level_match.group(1)
                aliases.append(f"level {level_match.group(1)}")
                aliases.append(f"niveau {level_match.group(1)}")
            if corridor_match:
                corridor = f"corridor {corridor_match.group(1)}"
                constraints["possible_zone"] = constraints.get("possible_zone") or corridor
                aliases.append(corridor)
            name = _remove_location_words(text)
            if name:
                constraints["target_name"] = constraints.get("target_name") or name
                aliases.append(name)
            hyps.append(GoalHypothesis(
                interpretation="Target appears to be a named place with level/corridor constraints.",
                evidence_needed=["directory/map for the named place", "level/floor indicator", "corridor sign"],
                confidence="medium",
            ))

        # 10) Number-only room: 314, 204.
        if not hyps:
            m = re.fullmatch(r"\d+[A-Za-z]*", text)
            if m:
                room = text.upper()
                constraints["possible_room"] = room
                aliases.extend([room, f"Room {room}"])
                if re.fullmatch(r"\d{3,4}[A-Za-z]?", room):
                    constraints["possible_floor"] = room[0]
                    hyps.append(GoalHypothesis(
                        interpretation=f"May be a full room number '{room}'. In some buildings the first digit may indicate Floor {room[0]}.",
                        evidence_needed=[f"door label {room}", f"corridor room range containing {room}", f"floor indicator {room[0]} if the building uses first-digit floor numbering"],
                        confidence="medium",
                    ))
                else:
                    hyps.append(GoalHypothesis(
                        interpretation=f"May be a full room/location label '{room}'.",
                        evidence_needed=[f"door label {room}", "directory listing", "corridor range sign"],
                        confidence="medium",
                    ))

        # 11) Compact alphanumeric code: E36.
        if not hyps:
            m = re.fullmatch(r"([A-Za-z]+)(\d+[A-Za-z]*)", text)
            if m:
                zone, number = m.group(1).upper(), m.group(2).upper()
                constraints["possible_zone"] = zone
                constraints["possible_room"] = number
                aliases.extend([f"{zone}{number}", f"{zone}-{number}", f"Room {zone}{number}", f"Unit {zone}{number}"])
                hyps.append(GoalHypothesis(
                    interpretation=f"May be a full room/unit label '{zone}{number}', or Zone {zone} with unit/room {number}.",
                    evidence_needed=[f"door label {zone}{number}", f"zone/block sign {zone}", f"unit/room label {number}"],
                    confidence="medium",
                ))

        # 12) Named place fallback.
        if not hyps:
            target_type = "facility" if _looks_like_facility(text) else "room_or_location"
            constraints["target_name"] = text
            aliases.append(text)
            hyps.append(GoalHypothesis(
                interpretation=f"Target may be a named office, department, lab, service, or facility: '{text}'.",
                evidence_needed=["directory/map", "department or facility sign", "official front-desk/staff directions"],
                confidence="medium",
            ))

        aliases = _unique_strings(_clean_aliases(aliases))
        tokens = _unique_strings(tokens + aliases)

        return NavigationGoal(
            raw_goal=text,
            target_type=target_type,
            known_tokens=tokens,
            aliases=aliases,
            constraints=constraints,
            possible_interpretations=hyps,
            search_strategy=_build_search_strategy(text, aliases, constraints, target_type),
        )


def _empty_constraints() -> dict[str, Any]:
    return {
        "possible_building": None,
        "possible_floor": None,
        "possible_room": None,
        "possible_zone": None,
        "target_name": None,
    }


def _merge_constraints(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = _empty_constraints()
    merged.update(base or {})
    for key in merged:
        value = override.get(key)
        if value not in (None, "", [], {}):
            merged[key] = value
    return merged


def _extract_tokens(text: str) -> list[str]:
    # Preserve exact goal first, then useful alphanumeric/location chunks.
    rough = re.findall(
    r"[A-Za-z]+\d*[A-Za-z]*\+?|\d+[A-Za-z]*|[A-Za-z]*\d+[.\-_]\d+[A-Za-z]*|\d+[.]\d+",
    text,
    )
    # Also keep common multi-word location phrases.
    phrases = []
    for pat in (r"niveau\s*\d+", r"level\s*\d+", r"floor\s*\d+", r"corridor\s*\d+[A-Za-z]*", r"gate\s*[A-Za-z]*\d+", r"suite\s*[A-Za-z]*\d+[A-Za-z]*", r"room\s*[A-Za-z]*\d+[A-Za-z]*"):
        phrases.extend(re.findall(pat, text, flags=re.I))
    return _unique_strings([text] + phrases + rough)

def _remove_location_words(text: str) -> str:
    cleaned = re.sub(r"\b(niveau|level|floor)\s*\d+\b", "", text, flags=re.I)
    cleaned = re.sub(r"\bcorridor\s*\d+[A-Za-z]*\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;-_")
    return cleaned


def _looks_like_facility(text: str) -> bool:
    return bool(re.search(r"\b(reception|library|office|lab|clinic|pharmacy|apotheek|department|desk|toilet|restroom|cafe|canteen)\b", text, re.I))


def _clean_aliases(aliases: list[Any]) -> list[str]:
    cleaned: list[str] = []
    for item in aliases:
        if item is None:
            continue
        s = str(item).strip()
        if not s:
            continue
        # Remove duplicate spaces only; do not remove zeros or punctuation because door labels may use them.
        cleaned.append(re.sub(r"\s+", " ", s))
    return cleaned


def _normalize_confidence(value: Any) -> str:
    s = str(value).strip().lower()
    return s if s in {"high", "medium", "low"} else "medium"


def _build_search_strategy(text: str, aliases: list[str], constraints: dict[str, Any], target_type: str) -> str:
    labels = aliases[:4] or [text]
    if target_type in {"department", "facility", "lab"} or constraints.get("target_name"):
        return (
            f"Search for directory/map or signs matching {labels}; then use floor, corridor, zone, "
            "and official front-desk/staff evidence to refine the route."
        )
    return (
        f"Search for door labels, corridor range signs, directories, or floor/zone indicators matching {labels}; "
        "treat building/floor/room structure as a hypothesis until confirmed."
    )


def _unique_strings(items: list[Any]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        if item is None:
            continue
        s = str(item).strip()
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            out.append(s)
            seen.add(key)
    return out


def _extract_json(text: str) -> dict[str, Any] | None:
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None