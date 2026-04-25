"""
Cross-Examination Arena — Case generator.

Three fully-fleshed templates (alibi, knowledge denial, possession denial),
six slot-fill pools, and the public `generate_case()` factory.

Add 2 more templates on Day 1 if time allows. Each new template just needs
to return a dict with the same shape as `_make_case_dict` produces.
"""

import random
from dataclasses import dataclass
from typing import Dict, List


# ---------------------------------------------------------------------------
# Contradiction dataclass (mirror of witness.Contradiction so the case file
# stays self-contained; in your real codebase, import from witness.py).
# ---------------------------------------------------------------------------
@dataclass
class Contradiction:
    cid: str
    trigger_keywords: List[str]
    sealed_claim: str
    disprover_evidence_id: str
    triggered: bool = False
    surfaced: bool = False


# ---------------------------------------------------------------------------
# Slot-fill pools. Extend these freely — each entry adds case variety.
# ---------------------------------------------------------------------------
NAMES = [
    "Sara Patel", "Mark Chen", "Alex Reyes", "Priya Sharma", "Jordan Kim",
    "Diana Okafor", "Tomas Ruiz", "Yelena Volkov", "Hassan Ali", "Mei Tanaka",
    "Carlos Mendes", "Ingrid Lindqvist", "Dev Krishnan", "Naomi Brown",
    "Samir Haque", "Lila Andersson", "Theo Morales", "Reza Farahani",
    "Kavita Iyer", "Owen Nakamura", "Esme Dubois", "Ravi Joshi",
    "Hana Lee", "Felix Bauer", "Aaliyah Singh", "Mateo Silva",
    "Zoe Whitfield", "Daichi Sato", "Nadine Mahfouz", "Pavel Novak",
]

PLACES = [
    "the Riverside Cafe on Fifth Avenue",
    "the parking garage at 22 Elm Street",
    "the Marriott hotel lobby downtown",
    "the office of Henderson & Co. on Pine",
    "the warehouse on Industrial Drive",
    "the gym on Lexington",
    "Brennan's Bar near Union Square",
    "the apartment at 480 Oak Lane",
    "the boardwalk at Crescent Beach",
    "the Greyhound bus station on Central",
    "the parking lot behind the Westgate Mall",
    "the rooftop bar at the Stratton Hotel",
    "the public library on Walnut Street",
    "the diner at the corner of 9th and Broad",
]

TIMES = [
    "8:15 PM", "9:00 PM", "9:45 PM", "10:30 PM", "11:00 PM", "11:45 PM",
    "12:15 AM", "1:00 AM", "7:30 PM", "6:45 PM",
]

DATES = [
    "March 14th", "April 3rd", "January 22nd", "October 17th", "August 9th",
    "December 5th", "May 28th", "June 14th",
]

OBJECTS = [
    "the .38 revolver", "the briefcase", "the Rolex watch", "the burner phone",
    "the laptop", "the security badge", "the diamond pendant",
    "the warehouse keys", "the master keycard",
]

MOTIVES = [
    "a $40,000 unpaid debt",
    "a contested inheritance",
    "a botched business partnership",
    "an extramarital affair that had been discovered",
    "a longstanding professional rivalry",
    "the dismissal from a senior position six months earlier",
]


# ---------------------------------------------------------------------------
# Helper to build the case dict in a consistent shape.
# ---------------------------------------------------------------------------
def _make_case_dict(case_id, brief, ground_truth, story, evidence, contradictions):
    return {
        "case_id": case_id,
        "case_brief": brief,
        "ground_truth": ground_truth,
        "witness_story": story,
        "evidence": evidence,
        "contradictions": contradictions,
    }


def _case_suffix() -> str:
    """Seed-respecting six-character case suffix."""
    return f"{random.randint(0, 16**6 - 1):06x}"


# ---------------------------------------------------------------------------
# Helper functions for time manipulation
# ---------------------------------------------------------------------------
def _minus_minutes(time_str, minutes):
    # Simple helper for time arithmetic
    hour, minute = map(int, time_str.replace(" PM", "").replace(" AM", "").split(":"))
    if "PM" in time_str and hour != 12:
        hour += 12
    total_minutes = hour * 60 + minute - minutes
    new_hour = total_minutes // 60
    new_minute = total_minutes % 60
    return f"{new_hour % 12 or 12}:{new_minute:02d} {'PM' if new_hour >= 12 else 'AM'}"

def _plus_minutes(time_str, minutes):
    hour, minute = map(int, time_str.replace(" PM", "").replace(" AM", "").split(":"))
    if "PM" in time_str and hour != 12:
        hour += 12
    total_minutes = hour * 60 + minute + minutes
    new_hour = total_minutes // 60
    new_minute = total_minutes % 60
    return f"{new_hour % 12 or 12}:{new_minute:02d} {'PM' if new_hour >= 12 else 'AM'}"


# ---------------------------------------------------------------------------
# Template 1 — Alibi conflict
# ---------------------------------------------------------------------------
def generate_alibi_case():
    """Witness claims they were at place A; phone/transit data places them at B."""
    perp = random.choice(NAMES)
    fake_place = random.choice(PLACES)
    real_place = random.choice([p for p in PLACES if p != fake_place])
    crime_place = random.choice([p for p in PLACES if p not in (fake_place, real_place)])
    t = random.choice(TIMES)
    date = random.choice(DATES)
    motive = random.choice(MOTIVES)

    case_id = f"alibi_{_case_suffix()}"
    brief = (
        f"{perp} is the prime suspect in the burglary at {crime_place} on the night of {date}. "
        f"The state alleges the burglary occurred at approximately {t}. "
        f"The defendant has taken the stand."
    )
    ground_truth = (
        f"{perp} was at {real_place} at {t} on {date}, then traveled to {crime_place} "
        f"and committed the burglary. Motive: {motive}."
    )
    story = {
        "alibi":         f"I was at {fake_place} that entire evening with friends.",
        "where":         f"I told you, {fake_place}. I never left.",
        "evening":       f"I had dinner and drinks at {fake_place} until close.",
        "warehouse":     f"I have never been to {crime_place}. Not once.",
        "crime":         f"I had nothing to do with what happened at {crime_place}.",
        "motive":        f"I had no reason to do anything to anyone.",
        "victim":        f"We had a professional disagreement, that's all. Nothing serious.",
        "phone":         f"My phone was with me the whole night.",
    }
    evidence = {
        "phone_tower_log":
            f"Cellular records: {perp}'s phone connected to the tower nearest "
            f"{real_place} at {t}, then to the tower covering {crime_place} 22 minutes later.",
        "ride_receipt":
            f"Rideshare receipt in {perp}'s account: dropoff at {real_place} at "
            f"{_minus_minutes(t, 30)}; second pickup at {real_place}, dropoff at "
            f"{crime_place} at {_plus_minutes(t, 18)}.",
        "victim_letter":
            f"Letter from the victim's attorney to {perp}, dated three weeks before the "
            f"burglary, formally demanding payment related to {motive}.",
        "doormat_print":
            f"A partial shoe print recovered from the doormat at {crime_place}, "
            f"matching size and tread of footwear later seized from {perp}'s residence.",
    }
    contradictions = [
        Contradiction(
            cid="c1_alibi_location",
            trigger_keywords=[
                "where were you", "alibi", "that night", "evening", "your location",
                t.lower(),
            ],
            sealed_claim=f"I was at {fake_place} the entire evening — I never left.",
            disprover_evidence_id="phone_tower_log",
        ),
        Contradiction(
            cid="c2_warehouse_familiarity",
            trigger_keywords=[
                "ever been to", "familiar with", "the warehouse", "the scene",
                "crime scene", crime_place.split()[1].lower() if len(crime_place.split()) > 1 else "warehouse",
            ],
            sealed_claim=f"I have never set foot at {crime_place}. I wouldn't even know how to get there.",
            disprover_evidence_id="ride_receipt",
        ),
        Contradiction(
            cid="c3_motive_denial",
            trigger_keywords=[
                "motive", "reason", "grievance", "dispute", "the victim",
                "any conflict", "disagreement",
            ],
            sealed_claim=f"I had no real grievance with the victim. We had a polite professional relationship.",
            disprover_evidence_id="victim_letter",
        ),
    ]
    return _make_case_dict(case_id, brief, ground_truth, story, evidence, contradictions)


# ---------------------------------------------------------------------------
# Template 2 — Knowledge denial
# ---------------------------------------------------------------------------
def generate_knowledge_denial_case():
    """Witness denies knowing victim/place/object; documentary evidence shows otherwise."""
    perp = random.choice(NAMES)
    victim = random.choice([n for n in NAMES if n != perp])
    place = random.choice(PLACES)
    date = random.choice(DATES)

    case_id = f"knowledge_{_case_suffix()}"
    brief = (
        f"{perp} is charged with embezzlement from {place}. The state alleges {perp} "
        f"systematically diverted funds in coordination with the manager, {victim}, "
        f"between {date} and the present. The defendant denies knowing {victim}."
    )
    ground_truth = (
        f"{perp} and {victim} have been business associates for over two years. "
        f"They met at {place} and collaborated on the embezzlement scheme."
    )
    story = {
        "know":          f"I don't know {victim} at all. Never met them.",
        "victim":        f"Who is {victim}? I've never heard of them.",
        "place":         f"I've never been to {place}.",
        "embezzlement":  f"I had nothing to do with any embezzlement.",
        "work":          f"I work in a completely different field.",
        "meeting":       f"I've never met with anyone about this.",
    }
    evidence = {
        "email_chain":
            f"Email chain between {perp} and {victim} discussing the embezzlement plan, "
            f"dated {date}, sent from {perp}'s work email.",
        "meeting_notes":
            f"Meeting notes from {place} conference room, dated two weeks ago, "
            f"with {perp} and {victim}'s signatures.",
        "bank_records":
            f"Bank records showing transfers from {place}'s accounts to {perp}'s "
            f"personal account, totaling $150,000 over six months.",
        "visitor_log":
            f"Security log from {place} showing {perp}'s entry and exit times "
            f"on multiple dates, including meetings with {victim}.",
    }
    contradictions = [
        Contradiction(
            cid="c1_know_victim",
            trigger_keywords=[
                "know", "met", "familiar", victim.split()[0].lower(),
                "relationship", "acquaintance",
            ],
            sealed_claim=f"I have never met {victim} in my life.",
            disprover_evidence_id="email_chain",
        ),
        Contradiction(
            cid="c2_place_visit",
            trigger_keywords=[
                "been to", "visited", place.split()[1].lower() if len(place.split()) > 1 else "place",
                "location", "office",
            ],
            sealed_claim=f"I've never set foot in {place}.",
            disprover_evidence_id="visitor_log",
        ),
        Contradiction(
            cid="c3_collaboration",
            trigger_keywords=[
                "worked with", "collaborated", "together", "scheme",
                "embezzlement", "plan",
            ],
            sealed_claim=f"I worked alone. No one else was involved.",
            disprover_evidence_id="meeting_notes",
        ),
    ]
    return _make_case_dict(case_id, brief, ground_truth, story, evidence, contradictions)


# ---------------------------------------------------------------------------
# Template 3 — Possession denial
# ---------------------------------------------------------------------------
def generate_possession_denial_case():
    """Witness denies possessing object; evidence shows they did."""
    perp = random.choice(NAMES)
    obj = random.choice(OBJECTS)
    place = random.choice(PLACES)
    date = random.choice(DATES)
    motive = random.choice(MOTIVES)

    case_id = f"possession_{_case_suffix()}"
    brief = (
        f"{perp} is charged with theft of {obj} from {place} on {date}. "
        f"The state alleges {perp} stole {obj} and used it in a subsequent crime. "
        f"The defendant denies ever possessing {obj}."
    )
    ground_truth = (
        f"{perp} stole {obj} from {place} on {date}. They used it later that night "
        f"in a robbery. Motive for the theft: {motive}."
    )
    story = {
        "possession":   f"I've never had {obj}. Never seen it.",
        "object":       f"What is {obj}? I've never heard of it.",
        "theft":        f"I didn't steal anything.",
        "crime":        f"I had nothing to do with any crime.",
        "place":        f"I've never been to {place}.",
        "motive":       f"I had no reason to steal anything.",
    }
    evidence = {
        "surveillance_footage":
            f"Surveillance footage from {place} showing {perp} entering at 8:00 PM "
            f"on {date}, taking {obj}, and leaving at 8:15 PM.",
        "fingerprint_analysis":
            f"Fingerprint analysis of {obj} recovered from the crime scene, "
            f"matching {perp}'s fingerprints on file.",
        "witness_statement":
            f"Statement from an employee at {place} identifying {perp} as the person "
            f"who took {obj} on {date}.",
        "pawn_receipt":
            f"Pawn shop receipt in {perp}'s name, dated the day after {date}, "
            f"for selling an item matching the description of {obj}.",
    }
    contradictions = [
        Contradiction(
            cid="c1_possession_object",
            trigger_keywords=[
                "have", "possessed", "owned", obj.split()[1].lower() if len(obj.split()) > 1 else "object",
                "in your possession",
            ],
            sealed_claim=f"I've never possessed {obj} in my life.",
            disprover_evidence_id="fingerprint_analysis",
        ),
        Contradiction(
            cid="c2_theft_admission",
            trigger_keywords=[
                "stole", "took", "theft", "place", place.split()[1].lower() if len(place.split()) > 1 else "place",
            ],
            sealed_claim=f"I didn't steal anything from {place}.",
            disprover_evidence_id="surveillance_footage",
        ),
        Contradiction(
            cid="c3_disposal",
            trigger_keywords=[
                "sold", "pawned", "got rid of", "disposed", "object",
            ],
            sealed_claim=f"I don't know what happened to {obj} after it was stolen.",
            disprover_evidence_id="pawn_receipt",
        ),
    ]
    return _make_case_dict(case_id, brief, ground_truth, story, evidence, contradictions)


# ---------------------------------------------------------------------------
# Template 4 — Timeline shift (new template)
# ---------------------------------------------------------------------------
def generate_timeline_shift_case():
    """Witness claims event happened at time A; evidence shows time B."""
    perp = random.choice(NAMES)
    place = random.choice(PLACES)
    fake_time = random.choice(TIMES)
    real_time = random.choice([t for t in TIMES if t != fake_time])
    date = random.choice(DATES)
    motive = random.choice(MOTIVES)

    case_id = f"timeline_{_case_suffix()}"
    brief = (
        f"{perp} is accused of assault at {place} on {date}. The victim claims the assault "
        f"occurred at {real_time}. The defendant claims they were elsewhere at that time."
    )
    ground_truth = (
        f"{perp} assaulted the victim at {place} at {real_time} on {date}. "
        f"They falsely claimed to be at another location at {fake_time}. Motive: {motive}."
    )
    story = {
        "time":         f"The assault happened at {fake_time}, not {real_time}.",
        "when":         f"It was around {fake_time}. I remember clearly.",
        "alibi":        f"I was at home at {fake_time}. I have witnesses.",
        "assault":      f"I didn't assault anyone.",
        "place":        f"I've never been to {place}.",
        "motive":       f"I had no reason to hurt anyone.",
    }
    evidence = {
        "victim_statement":
            f"Victim's statement: The assault occurred at {real_time} on {date} at {place}.",
        "surveillance_timestamp":
            f"Surveillance footage timestamped at {real_time} showing {perp} at {place}.",
        "phone_records":
            f"Phone records showing {perp}'s location at {place} at {real_time}.",
        "witness_alibi":
            f"Alibi witness statement claiming {perp} was with them at {fake_time}, "
            f"but the witness later recanted under pressure.",
        "motive_email":
            f"Email from {perp} sent the week before {date} describing {motive} "
            f"and threatening to confront the victim at {place}.",
    }
    contradictions = [
        Contradiction(
            cid="c1_time_of_assault",
            trigger_keywords=[
                "time", "when", "occurred", "happened", real_time.lower(),
            ],
            sealed_claim=f"The assault happened at {fake_time}, not {real_time}.",
            disprover_evidence_id="surveillance_timestamp",
        ),
        Contradiction(
            cid="c2_location_during_assault",
            trigger_keywords=[
                "where", "location", "place", place.split()[1].lower() if len(place.split()) > 1 else "place",
            ],
            sealed_claim=f"I was not at {place} during the assault.",
            disprover_evidence_id="phone_records",
        ),
        Contradiction(
            cid="c3_motive_denial",
            trigger_keywords=[
                "motive", "reason", "why", "conflict", "grievance",
            ],
            sealed_claim="I had no conflict with the victim and no reason to confront anyone.",
            disprover_evidence_id="motive_email",
        ),
    ]
    return _make_case_dict(case_id, brief, ground_truth, story, evidence, contradictions)


# ---------------------------------------------------------------------------
# Template 5 — Motive coverup (new template)
# ---------------------------------------------------------------------------
def generate_motive_coverup_case():
    """Witness denies motive; evidence reveals hidden relationship."""
    perp = random.choice(NAMES)
    victim = random.choice([n for n in NAMES if n != perp])
    place = random.choice(PLACES)
    date = random.choice(DATES)
    motive = random.choice(MOTIVES)

    case_id = f"motive_{_case_suffix()}"
    brief = (
        f"{perp} is charged with harassment of {victim}. The state alleges {perp} "
        f"stalked {victim} at {place} on {date}. The defendant denies any motive."
    )
    ground_truth = (
        f"{perp} harassed {victim} due to {motive}. They had a hidden relationship "
        f"that went sour, leading to the stalking at {place} on {date}."
    )
    story = {
        "motive":       f"I had no reason to harass {victim}.",
        "relationship": f"I barely knew {victim}. Just an acquaintance.",
        "harassment":   f"I didn't harass anyone.",
        "place":        f"I've never been to {place}.",
        "victim":       f"{victim} and I had no issues.",
    }
    evidence = {
        "text_messages":
            f"Text messages between {perp} and {victim} showing romantic relationship "
            f"that ended badly due to {motive}.",
        "surveillance_stalking":
            f"Surveillance footage of {perp} following {victim} at {place} on {date}.",
        "police_reports":
            f"Previous police reports of {perp} contacting {victim} inappropriately.",
        "witness_statement":
            f"Witness statement from a mutual friend confirming the relationship and motive.",
        "parking_receipt":
            f"Parking receipt in {perp}'s name placing their car beside {place} "
            f"on {date}, fifteen minutes before {victim} arrived.",
    }
    contradictions = [
        Contradiction(
            cid="c1_relationship_denial",
            trigger_keywords=[
                "relationship", "knew", "acquaintance", victim.split()[0].lower(),
            ],
            sealed_claim=f"I barely knew {victim}. We were just casual acquaintances.",
            disprover_evidence_id="text_messages",
        ),
        Contradiction(
            cid="c2_motive_denial",
            trigger_keywords=[
                "motive", "reason", "why", "harass",
            ],
            sealed_claim=f"I had absolutely no motive to harass {victim}.",
            disprover_evidence_id="witness_statement",
        ),
        Contradiction(
            cid="c3_place_denial",
            trigger_keywords=[
                "where", "place", "visited", "been to", place.split()[1].lower() if len(place.split()) > 1 else "place",
            ],
            sealed_claim=f"I was nowhere near {place} on {date}.",
            disprover_evidence_id="parking_receipt",
        ),
    ]
    return _make_case_dict(case_id, brief, ground_truth, story, evidence, contradictions)


# ---------------------------------------------------------------------------
# Template 6 - Corporate fraud deposition
# ---------------------------------------------------------------------------
def generate_corporate_fraud_case():
    """Executive denies knowledge, approval, and motive in an internal fraud inquiry."""
    executive = random.choice(NAMES)
    analyst = random.choice([n for n in NAMES if n != executive])
    company = random.choice(["Northstar Biologics", "Meridian Freight", "Aster Cloud", "HelioGrid Energy"])
    t = random.choice(TIMES)
    date = random.choice(DATES)
    motive = random.choice(MOTIVES)

    case_id = f"corporate_{_case_suffix()}"
    brief = (
        f"{executive}, a senior executive at {company}, is questioned about revenue manipulation "
        f"recorded on {date}. The company claims the irregular entries were a clerical mistake."
    )
    ground_truth = (
        f"{executive} approved backdated revenue entries at {t} on {date}, after {analyst} "
        f"warned that the quarter would miss guidance. Motive: {motive}."
    )
    story = {
        "approval": "I never approved any backdated revenue entries.",
        "analyst": f"{analyst} handled routine reporting. I did not discuss adjustments with them.",
        "forecast": "I was not worried about the quarterly forecast.",
        "revenue": "Any revenue issue was a clerical mistake, not my decision.",
        "meeting": "There was no late meeting about accounting entries.",
    }
    evidence = {
        "approval_log":
            f"Accounting system log: {executive} approved backdated entries at {t} on {date}.",
        "analyst_chat":
            f"Chat from {analyst} to {executive}: 'Without the adjustment, we miss guidance by 8%.'",
        "board_deck":
            f"Draft board deck listing the missed forecast and {motive} as a risk to compensation.",
        "calendar_invite":
            f"Calendar invite: emergency revenue recognition meeting with {executive} and {analyst} at {t}.",
        "printer_log":
            "Printer maintenance log from the same floor, unrelated to the accounting entries.",
    }
    contradictions = [
        Contradiction(
            cid="c1_approval_denial",
            trigger_keywords=["approved", "approval", "signed off", "authorized", "revenue"],
            sealed_claim="I never approved or authorized any backdated revenue entries.",
            disprover_evidence_id="approval_log",
        ),
        Contradiction(
            cid="c2_analyst_discussion",
            trigger_keywords=["analyst", analyst.split()[0].lower(), "discuss", "guidance", "forecast"],
            sealed_claim=f"I did not discuss any forecast problem with {analyst}.",
            disprover_evidence_id="analyst_chat",
        ),
        Contradiction(
            cid="c3_late_meeting",
            trigger_keywords=["meeting", "calendar", "late", t.lower(), "emergency"],
            sealed_claim="There was no late meeting about accounting entries.",
            disprover_evidence_id="calendar_invite",
        ),
    ]
    return _make_case_dict(case_id, brief, ground_truth, story, evidence, contradictions)


# ---------------------------------------------------------------------------
# Template 7 - Workplace investigation
# ---------------------------------------------------------------------------
def generate_workplace_investigation_case():
    """Manager denies a message, location, and prior warning in an HR inquiry."""
    manager = random.choice(NAMES)
    employee = random.choice([n for n in NAMES if n != manager])
    place = random.choice(["Conference Room B", "the sixth-floor break room", "the loading dock", "the HR office"])
    date = random.choice(DATES)
    t = random.choice(TIMES)

    case_id = f"workplace_{_case_suffix()}"
    brief = (
        f"{manager} is questioned in an HR investigation after {employee} reported retaliation "
        f"at {place} on {date}. The manager denies sending threatening messages or being nearby."
    )
    ground_truth = (
        f"{manager} sent a threatening message before meeting {employee} at {place} at {t}. "
        f"HR had warned {manager} one week earlier."
    )
    story = {
        "message": f"I never sent {employee} any threatening message.",
        "where": f"I was not near {place} that day.",
        "warning": "HR never warned me about retaliation.",
        "employee": f"{employee} and I barely interacted.",
        "retaliation": "I did not retaliate against anyone.",
    }
    evidence = {
        "message_export":
            f"Company chat export: {manager} wrote to {employee}, 'Drop the complaint or regret it,' on {date}.",
        "badge_scan":
            f"Badge access log: {manager} entered {place} at {t} on {date}.",
        "hr_warning_memo":
            f"HR memo issued one week before {date}, warning {manager} not to contact {employee}.",
        "security_clip":
            f"Security clip still: {manager} standing outside {place} minutes before {employee} arrived.",
        "cafeteria_menu":
            "Cafeteria menu from that week, unrelated to the investigation.",
    }
    contradictions = [
        Contradiction(
            cid="c1_message_denial",
            trigger_keywords=["message", "text", "chat", "threat", employee.split()[0].lower()],
            sealed_claim=f"I never sent {employee} any threatening message.",
            disprover_evidence_id="message_export",
        ),
        Contradiction(
            cid="c2_location_denial",
            trigger_keywords=["where", "location", "near", "badge", place.split()[0].lower()],
            sealed_claim=f"I was nowhere near {place} on {date}.",
            disprover_evidence_id="badge_scan",
        ),
        Contradiction(
            cid="c3_warning_denial",
            trigger_keywords=["warning", "hr", "memo", "retaliation", "complaint"],
            sealed_claim="HR never warned me about contacting the employee.",
            disprover_evidence_id="hr_warning_memo",
        ),
    ]
    return _make_case_dict(case_id, brief, ground_truth, story, evidence, contradictions)


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------
CURRICULUM_DISTRIBUTIONS = {
    "easy": {"easy": 1.0},
    "warmup": {"easy": 1.0},
    "medium": {"easy": 0.25, "medium": 0.75},
    "main": {"easy": 0.2, "medium": 0.8},
    "hard": {"easy": 0.1, "medium": 0.3, "hard": 0.6},
    "challenge": {"medium": 0.25, "hard": 0.75},
    "mixed": {"easy": 0.3, "medium": 0.45, "hard": 0.25},
}


def _sample_difficulty(curriculum_stage: str, distribution: Dict[str, float] | None = None) -> str:
    choices = distribution or CURRICULUM_DISTRIBUTIONS.get(curriculum_stage, {curriculum_stage: 1.0})
    labels = list(choices.keys())
    weights = list(choices.values())
    return random.choices(labels, weights=weights, k=1)[0]


def _apply_difficulty(case: Dict, difficulty: str) -> Dict:
    contradictions = list(case["contradictions"])

    if difficulty == "easy":
        case["contradictions"] = contradictions[:1]
    elif difficulty == "medium":
        case["contradictions"] = contradictions[:2]
        for contradiction in case["contradictions"]:
            if "where" not in contradiction.trigger_keywords:
                contradiction.trigger_keywords.append("where")
    elif difficulty == "hard":
        case["contradictions"] = contradictions[: max(3, len(contradictions))]
        case["evidence"]["irrelevant_weather_report"] = (
            "Weather report: light rain near the courthouse. It does not address any witness claim."
        )
        case["evidence"]["unrelated_receipt"] = (
            "Receipt for coffee purchased by an unrelated bystander earlier that afternoon."
        )
        for contradiction in case["contradictions"]:
            if "why" not in contradiction.trigger_keywords:
                contradiction.trigger_keywords.append("why")
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    case["difficulty"] = difficulty
    return case


def generate_case(
    difficulty: str | None = None,
    curriculum_stage: str = "medium",
    distribution: Dict[str, float] | None = None,
) -> Dict:
    """
    Generate a random cross-examination case.

    Args:
        difficulty: "easy", "medium", or "hard" — controls number of contradictions.

    Returns:
        Case dict with brief, story, evidence, contradictions.
    """
    templates = [
        generate_alibi_case,
        generate_knowledge_denial_case,
        generate_possession_denial_case,
        generate_timeline_shift_case,
        generate_motive_coverup_case,
        generate_corporate_fraud_case,
        generate_workplace_investigation_case,
    ]
    selected_difficulty = difficulty or _sample_difficulty(curriculum_stage, distribution)
    return _apply_difficulty(random.choice(templates)(), selected_difficulty)


if __name__ == "__main__":
    # Test the generator
    case = generate_case()
    print(f"Case ID: {case['case_id']}")
    print(f"Brief: {case['case_brief']}")
    print(f"Contradictions: {len(case['contradictions'])}")
    for c in case["contradictions"]:
        print(f"  - {c.cid}: {c.trigger_keywords}")
