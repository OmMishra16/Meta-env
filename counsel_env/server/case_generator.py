"""
Cross-Examination Arena — case generator.

Five fully-fleshed templates with slot-fill pools. Each call to generate_case()
returns a fresh procgen case dict with seeded contradictions.
"""

import random
import uuid
from typing import Dict, List

try:
    from .witness import Contradiction
except ImportError:
    from witness import Contradiction  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Slot-fill pools
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
# Helper
# ---------------------------------------------------------------------------
def _make_case_dict(
    case_id: str,
    brief: str,
    ground_truth: str,
    story: Dict[str, str],
    evidence: Dict[str, str],
    contradictions: List[Contradiction],
) -> dict:
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
def generate_alibi_case() -> dict:
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
        "alibi":     f"I was at {fake_place} that entire evening with friends.",
        "where":     f"I told you, {fake_place}. I never left.",
        "evening":   f"I had dinner and drinks at {fake_place} until close.",
        "warehouse": f"I have never been to {crime_place}. Not once.",
        "crime":     f"I had nothing to do with what happened at {crime_place}.",
        "motive":    f"I had no reason to do anything to anyone.",
        "victim":    f"We had a professional disagreement, that's all. Nothing serious.",
        "phone":     f"My phone was with me the whole night.",
    }
    evidence = {
        "phone_tower_log": (
            f"Cellular records: {perp}'s phone connected to the tower nearest "
            f"{real_place} at {t}, then to the tower covering {crime_place} 22 minutes later."
        ),
        "ride_receipt": (
            f"Rideshare receipt in {perp}'s account: dropoff at {real_place} at "
            f"{_minus_minutes(t, 30)}; second pickup at {real_place}, dropoff at "
            f"{crime_place} at {_plus_minutes(t, 18)}."
        ),
        "victim_letter": (
            f"Letter from the victim's attorney to {perp}, dated three weeks before the "
            f"burglary, formally demanding payment related to {motive}."
        ),
        "doormat_print": (
            f"A partial shoe print recovered from the doormat at {crime_place}, "
            f"matching size and tread of footwear later seized from {perp}'s residence."
        ),
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
                "crime scene",
                crime_place.split()[1].lower() if len(crime_place.split()) > 1 else "warehouse",
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
def generate_knowledge_denial_case() -> dict:
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
        "victim":        f"I don't know anyone by the name {victim}. I've never met them.",
        "manager":       f"I have no idea who the manager of {place} is. Why would I?",
        "association":   f"I have absolutely no business relationship with {victim}.",
        "meeting":       f"I have never met {victim} in my life.",
        "communication": f"I have not communicated with {victim} — not by phone, email, or any other means.",
        "money":         "I have nothing to do with any missing funds.",
        "embezzlement":  f"I had no role in any financial wrongdoing at {place}.",
    }
    evidence = {
        "email_chain": (
            f"A 47-message email thread between {perp} and {victim} spanning eighteen months, "
            f"discussing 'the project,' invoices, payment routing, and offshore accounts."
        ),
        "shared_photo": (
            f"A photo recovered from {perp}'s phone, dated four months before the alleged "
            f"embezzlement began, showing {perp} and {victim} dining together at {place}."
        ),
        "wire_records": (
            f"Bank records showing {perp} received seven wire transfers from a shell company "
            f"controlled by {victim}, totaling $94,000."
        ),
        "calendar_entries": (
            f"Calendar export from {perp}'s work account listing 23 meetings with {victim} "
            f"over the past two years, eleven of them at {place}."
        ),
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
def generate_possession_denial_case() -> dict:
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
        "weapon":    f"I have never owned {obj}. I have never even held one.",
        "ownership": f"That is not mine. I don't own any such thing.",
        "purchase":  f"I have never purchased {obj}. I would have no reason to.",
        "scene":     f"I was at {place}, yes, but I had nothing on me.",
        "carry":     f"I was not carrying any object that night.",
        "permit":    f"I do not hold any kind of permit or license for {obj}.",
        "store":     "I have never been to a store that sells anything like that.",
    }
    evidence = {
        "purchase_record": (
            f"A point-of-sale receipt from Westend Sporting Goods, dated eight months "
            f"before the assault, showing {perp} purchased {obj} using their own credit card. "
            f"Signature on file."
        ),
        "permit_application": (
            f"A signed and notarized permit application filed by {perp} with the county "
            f"clerk, declaring legal ownership of {obj}, with serial number recorded."
        ),
        "social_media_photo": (
            f"An Instagram post from {perp}'s verified account, dated three months before "
            f"the assault, showing {perp} holding {obj} at a recreational range."
        ),
        "fingerprint_report": (
            f"A forensic report identifying three latent prints recovered from {obj}, "
            f"all matched to {perp} with high confidence."
        ),
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
# Template 4 — Timeline shift
# Witness claims a key event happened at T1; CCTV / receipts timestamp T2.
# ---------------------------------------------------------------------------
def generate_timeline_shift_case() -> dict:
    """Witness claims event at T1; CCTV/receipts show it happened at T2."""
    perp = random.choice(NAMES)
    victim = random.choice([n for n in NAMES if n != perp])
    place = random.choice(PLACES)
    date = random.choice(DATES)
    t_claimed = random.choice(TIMES)
    t_real = random.choice([t for t in TIMES if t != t_claimed])
    motive = random.choice(MOTIVES)

    case_id = f"timeline_{uuid.uuid4().hex[:6]}"
    brief = (
        f"{perp} is suspected of assaulting {victim} at {place} on {date}. "
        f"{perp} claims the altercation — which they admit took place — occurred at {t_claimed}, "
        f"when {victim} had already left. CCTV and receipt evidence suggest otherwise."
    )
    ground_truth = (
        f"The assault occurred at {t_real}, not {t_claimed}. At {t_real}, {victim} was "
        f"still present at {place}. {perp}'s claimed timeline is fabricated to create "
        f"an apparent non-overlap. Motive: {motive}."
    )
    story = {
        "time":      f"I arrived at {place} around {t_claimed}. That's when everything happened.",
        "when":      f"I've been very clear — it was {t_claimed}. I remember because I checked my watch.",
        "victim":    f"By {t_claimed}, {victim} had already left. I never saw them that night.",
        "argument":  f"There was no argument. I was calm the entire time.",
        "left":      f"I left {place} by {_plus_minutes(t_claimed, 20)} at the latest.",
        "cctv":      f"I'm not aware of any cameras at {place}.",
        "receipt":   f"I didn't buy anything that night. I had no reason to.",
    }
    evidence = {
        "cctv_timestamp": (
            f"CCTV footage from {place} timestamped {t_real}, showing {perp} and "
            f"{victim} in a heated confrontation. Metadata is authenticated."
        ),
        "bar_receipt": (
            f"A credit card receipt from {place} showing {perp} purchased drinks at "
            f"{_minus_minutes(t_real, 10)}, placing {perp} on-site at the time of the assault."
        ),
        "witness_statement": (
            f"A signed statement from the bartender at {place} confirming that both "
            f"{perp} and {victim} were present until approximately {_plus_minutes(t_real, 15)}."
        ),
        "motive_doc": (
            f"Court records documenting a civil judgment against {perp} in favor of "
            f"{victim} related to {motive}, finalized two months before the assault."
        ),
    }
    contradictions = [
        Contradiction(
            cid="c1_time_of_arrival",
            trigger_keywords=[
                "what time", "when did you arrive", "time of", t_claimed.lower(),
                "that evening", "clock", "watch",
            ],
            sealed_claim=f"I arrived at {place} at {t_claimed} and that is when everything happened.",
            disprover_evidence_id="cctv_timestamp",
        ),
        Contradiction(
            cid="c2_victim_presence",
            trigger_keywords=[
                "was the victim", "did you see", "was anyone else", "victim still there",
                victim.split()[0].lower(),
            ],
            sealed_claim=f"By {t_claimed}, {victim} had already left {place}. I did not see them.",
            disprover_evidence_id="witness_statement",
        ),
        Contradiction(
            cid="c3_no_purchase",
            trigger_keywords=[
                "purchase", "buy anything", "receipt", "pay", "order",
            ],
            sealed_claim=f"I did not purchase anything at {place} that night.",
            disprover_evidence_id="bar_receipt",
        ),
    ]
    return _make_case_dict(case_id, brief, ground_truth, story, evidence, contradictions)


# ---------------------------------------------------------------------------
# Template 5 — Motive coverup
# Witness denies any grievance with victim; emails/letters prove a serious dispute.
# ---------------------------------------------------------------------------
def generate_motive_coverup_case() -> dict:
    """Witness denies grievance with victim; emails and letters document a major dispute."""
    perp = random.choice(NAMES)
    victim = random.choice([n for n in NAMES if n != perp])
    place = random.choice(PLACES)
    date = random.choice(DATES)
    motive = random.choice(MOTIVES)
    obj = random.choice(OBJECTS)

    case_id = f"motive_{uuid.uuid4().hex[:6]}"
    brief = (
        f"{perp} is charged with the theft of {obj} belonging to {victim} at {place} on {date}. "
        f"{perp} acknowledges knowing {victim} but denies having any grievance, "
        f"claiming the relationship was entirely cordial."
    )
    ground_truth = (
        f"{perp} and {victim} were embroiled in a serious dispute over {motive}. "
        f"{perp} stole {obj} as retaliation. The 'cordial relationship' claim is fabricated."
    )
    story = {
        "relationship": f"My relationship with {victim} was perfectly professional. No issues whatsoever.",
        "dispute":      f"There was no dispute between us. That's simply not true.",
        "grievance":    f"I had absolutely no grievance against {victim}.",
        "money":        f"There was no financial disagreement between {victim} and me.",
        "argument":     f"We never argued. Not once. We were on good terms.",
        "threat":       f"I never threatened {victim}. That is a complete fabrication.",
        "motive":       f"I had no reason whatsoever to take anything from {victim}.",
    }
    evidence = {
        "threatening_email": (
            f"An email from {perp} to {victim} dated six weeks before the theft, "
            f"with subject line 'Final Warning,' containing explicit threats over {motive}."
        ),
        "legal_correspondence": (
            f"A certified letter from {perp}'s attorney to {victim} demanding resolution "
            f"of the dispute over {motive}, sent three months before the incident."
        ),
        "text_messages": (
            f"A series of 18 text messages between {perp} and {victim} over the final "
            f"two weeks before the theft, escalating in hostility and referencing {obj}."
        ),
        "witness_account": (
            f"A sworn affidavit from a mutual colleague who witnessed a public argument "
            f"between {perp} and {victim} at {place} about {motive}, one month prior."
        ),
    }
    contradictions = [
        Contradiction(
            cid="c1_no_grievance",
            trigger_keywords=[
                "grievance", "dispute", "argument", "problem with", "issue with",
                "disagreement", "conflict",
            ],
            sealed_claim=f"I had no grievance with {victim}. Our relationship was entirely professional.",
            disprover_evidence_id="threatening_email",
        ),
        Contradiction(
            cid="c2_cordial_relationship",
            trigger_keywords=[
                "relationship", "how would you describe", "friendly", "cordial",
                "get along", "feelings about", victim.split()[0].lower(),
            ],
            sealed_claim=f"My relationship with {victim} was cordial. There was no hostility.",
            disprover_evidence_id="witness_account",
        ),
        Contradiction(
            cid="c3_no_legal_dispute",
            trigger_keywords=[
                "legal", "lawyer", "attorney", "lawsuit", "letter", "formal",
                "demand", "court",
            ],
            sealed_claim=f"There was no legal dispute between {victim} and me. No lawyers involved.",
            disprover_evidence_id="legal_correspondence",
        ),
    ]
    return _make_case_dict(case_id, brief, ground_truth, story, evidence, contradictions)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def _minus_minutes(timestr: str, minutes: int) -> str:
    return _shift_minutes(timestr, -minutes)


def _plus_minutes(timestr: str, minutes: int) -> str:
    return _shift_minutes(timestr, minutes)


def _shift_minutes(timestr: str, minutes: int) -> str:
    import datetime as _dt
    fmt = "%I:%M %p"
    t = _dt.datetime.strptime(timestr, fmt) + _dt.timedelta(minutes=minutes)
    return t.strftime(fmt).lstrip("0")


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
TEMPLATES = [
    generate_alibi_case,
    generate_knowledge_denial_case,
    generate_possession_denial_case,
    generate_timeline_shift_case,
    generate_motive_coverup_case,
]


def generate_case() -> dict:
    """Return one fully-populated case ready for the witness and environment."""
    return random.choice(TEMPLATES)()


# ---------------------------------------------------------------------------
# Smoke check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import dataclasses
    import json

    for _ in range(3):
        case = generate_case()
        case_out = {
            **{k: v for k, v in case.items() if k != "contradictions"},
            "contradictions": [dataclasses.asdict(c) for c in case["contradictions"]],
        }
        print(json.dumps(case_out, indent=2))
        print("---")
