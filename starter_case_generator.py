"""
Cross-Examination Arena — Starter case generator.

Drop this in at counsel_env/server/case_generator.py.

Three fully-fleshed templates (alibi, knowledge denial, possession denial),
six slot-fill pools, and the public `generate_case()` factory.

Add 2-4 more templates on Day 1 if time allows. Each new template just needs
to return a dict with the same shape as `_make_case_dict` produces.
"""

import random
import uuid
from dataclasses import dataclass, field
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

    case_id = f"alibi_{uuid.uuid4().hex[:6]}"
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

    case_id = f"knowledge_{uuid.uuid4().hex[:6]}"
    brief = (
        f"{perp} is charged with embezzlement from {place}. The state alleges {perp} "
        f"systematically diverted funds in coordination with the manager, {victim}, "
        f"between {date} and the present. The defendant denies knowing {victim}."
    )
    ground_truth = (
        f"{perp} and {victim} have been business associates for over two years. "
        f"They co-orchestrated the diversion of approximately $180,000 from {place}."
    )
    story = {
        "victim":      f"I don't know anyone by the name {victim}. I've never met them.",
        "manager":     f"I have no idea who the manager of {place} is. Why would I?",
        "association": f"I have absolutely no business relationship with {victim}.",
        "meeting":     f"I have never met {victim} in my life.",
        "communication": f"I have not communicated with {victim} — not by phone, email, or any other means.",
        "money":       "I have nothing to do with any missing funds.",
        "embezzlement": f"I had no role in any financial wrongdoing at {place}.",
    }
    evidence = {
        "email_chain":
            f"A 47-message email thread between {perp} and {victim} spanning eighteen months, "
            f"discussing 'the project,' invoices, payment routing, and offshore accounts.",
        "shared_photo":
            f"A photo recovered from {perp}'s phone, dated four months before the alleged "
            f"embezzlement began, showing {perp} and {victim} dining together at {place}.",
        "wire_records":
            f"Bank records showing {perp} received seven wire transfers from a shell company "
            f"controlled by {victim}, totaling $94,000.",
        "calendar_entries":
            f"Calendar export from {perp}'s work account listing 23 meetings with {victim} "
            f"over the past two years, eleven of them at {place}.",
    }
    contradictions = [
        Contradiction(
            cid="c1_knows_victim",
            trigger_keywords=[
                "do you know", "ever met", "acquainted with",
                victim.split()[0].lower(), victim.split()[1].lower(),
            ],
            sealed_claim=f"I have never met {victim}. I do not know that person.",
            disprover_evidence_id="shared_photo",
        ),
        Contradiction(
            cid="c2_no_communication",
            trigger_keywords=[
                "communicate", "email", "message", "speak", "contact", "correspond",
            ],
            sealed_claim=f"I have never communicated with {victim} in any form.",
            disprover_evidence_id="email_chain",
        ),
        Contradiction(
            cid="c3_no_business",
            trigger_keywords=[
                "business", "money", "financial", "transaction", "transfer", "payment",
                "wire",
            ],
            sealed_claim=f"There has never been any business or financial relationship between us.",
            disprover_evidence_id="wire_records",
        ),
    ]
    return _make_case_dict(case_id, brief, ground_truth, story, evidence, contradictions)


# ---------------------------------------------------------------------------
# Template 3 — Possession denial
# ---------------------------------------------------------------------------
def generate_possession_denial_case():
    """Witness denies owning a specific object; registration / photos prove ownership."""
    perp = random.choice(NAMES)
    obj = random.choice(OBJECTS)
    place = random.choice(PLACES)
    date = random.choice(DATES)

    case_id = f"possession_{uuid.uuid4().hex[:6]}"
    brief = (
        f"{perp} is being questioned in connection with the assault that occurred at "
        f"{place} on {date}. The state believes the assault was committed using {obj}, "
        f"which was recovered from the scene."
    )
    ground_truth = (
        f"{perp} legally purchased {obj} eight months ago, carried it on the night of "
        f"the assault, and dropped it at {place}. {perp} is denying ownership entirely."
    )
    story = {
        "weapon":      f"I have never owned {obj}. I have never even held one.",
        "ownership":   f"That is not mine. I don't own any such thing.",
        "purchase":    f"I have never purchased {obj}. I would have no reason to.",
        "scene":       f"I was at {place}, yes, but I had nothing on me.",
        "carry":       f"I was not carrying any object that night.",
        "permit":      f"I do not hold any kind of permit or license for {obj}.",
        "store":       "I have never been to a store that sells anything like that.",
    }
    evidence = {
        "purchase_record":
            f"A point-of-sale receipt from Westend Sporting Goods, dated eight months "
            f"before the assault, showing {perp} purchased {obj} using their own credit card. "
            f"Signature on file.",
        "permit_application":
            f"A signed and notarized permit application filed by {perp} with the county "
            f"clerk, declaring legal ownership of {obj}, with serial number recorded.",
        "social_media_photo":
            f"An Instagram post from {perp}'s verified account, dated three months before "
            f"the assault, showing {perp} holding {obj} at a recreational range.",
        "fingerprint_report":
            f"A forensic report identifying three latent prints recovered from {obj}, "
            f"all matched to {perp} with high confidence.",
    }
    contradictions = [
        Contradiction(
            cid="c1_ownership",
            trigger_keywords=[
                "own", "yours", "possession", "belong",
                obj.split()[-1].lower(),
            ],
            sealed_claim=f"I have never owned {obj}. It is not mine.",
            disprover_evidence_id="purchase_record",
        ),
        Contradiction(
            cid="c2_permit",
            trigger_keywords=[
                "permit", "license", "registered", "registration", "legally",
            ],
            sealed_claim=f"I have never applied for a permit for {obj}.",
            disprover_evidence_id="permit_application",
        ),
        Contradiction(
            cid="c3_handling",
            trigger_keywords=[
                "held", "handled", "touched", "ever fired", "ever used",
            ],
            sealed_claim=f"I have never held or touched {obj} in my life.",
            disprover_evidence_id="social_media_photo",
        ),
    ]
    return _make_case_dict(case_id, brief, ground_truth, story, evidence, contradictions)


# ---------------------------------------------------------------------------
# Time-arithmetic helpers (rough, good enough for narrative timestamps).
# ---------------------------------------------------------------------------
def _minus_minutes(timestr: str, minutes: int) -> str:
    return _shift_minutes(timestr, -minutes)


def _plus_minutes(timestr: str, minutes: int) -> str:
    return _shift_minutes(timestr, minutes)


def _shift_minutes(timestr: str, minutes: int) -> str:
    """Shift a '9:45 PM'-style time by N minutes. Crude but readable."""
    import datetime as _dt
    fmt = "%I:%M %p"
    t = _dt.datetime.strptime(timestr, fmt) + _dt.timedelta(minutes=minutes)
    return t.strftime(fmt).lstrip("0")


# ---------------------------------------------------------------------------
# Public factory.
# ---------------------------------------------------------------------------
TEMPLATES = [
    generate_alibi_case,
    generate_knowledge_denial_case,
    generate_possession_denial_case,
    # Add more templates here on Day 1 if time:
    #   generate_timeline_shift_case,
    #   generate_motive_coverup_case,
    #   generate_witness_chain_case,
    #   generate_inconsistent_memory_case,
]


def generate_case() -> dict:
    """Return one fully-populated case ready for the witness/environment."""
    return random.choice(TEMPLATES)()


# ---------------------------------------------------------------------------
# Smoke test — run this file directly to inspect 3 generated cases.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    for _ in range(3):
        case = generate_case()
        # Convert Contradiction dataclasses to dict for printing.
        case = {**case, "contradictions": [c.__dict__ for c in case["contradictions"]]}
        print(json.dumps(case, indent=2))
        print("---")
