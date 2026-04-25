"""Local validation for Counsel-Env without starting the OpenEnv server."""

from counsel_env.models import CounselAction, CounselObservation, CounselState
from counsel_env.server.case_generator import generate_case
from counsel_env.server.counsel_env_environment import CounselEnvironment
from counsel_env.server.rubrics import Total
from counsel_env.server.witness import Contradiction, Witness


def ok(message: str) -> None:
    print(f"[OK] {message}")


def main() -> None:
    print("=" * 60)
    print("COUNSEL-ENV VALIDATION TEST SUITE")
    print("=" * 60)

    print("\n[TEST 1] Case Generator")
    easy = generate_case(difficulty="easy")
    medium = generate_case(difficulty="medium")
    hard = generate_case(difficulty="hard")
    assert len(easy["contradictions"]) == 1
    assert len(medium["contradictions"]) == 2
    assert len(hard["contradictions"]) >= 3
    assert "irrelevant_weather_report" in hard["evidence"]
    ok(f"Generated easy/medium/hard cases: {easy['case_id']}, {medium['case_id']}, {hard['case_id']}")

    print("\n[TEST 2] Witness Logic")
    contradiction = Contradiction(
        cid="c1",
        trigger_keywords=["where were you"],
        sealed_claim="I was at the diner.",
        disprover_evidence_id="phone_log",
    )
    witness = Witness({}, [contradiction])
    assert witness.react_to_evidence("phone_log").endswith("exhibit.")
    assert witness.respond_to_question("Where were you?") == contradiction.sealed_claim
    assert witness.react_to_evidence("phone_log").startswith("[Witness stammers]")
    ok("Trigger -> sealed claim -> evidence surface works")

    print("\n[TEST 3] Environment Reward")
    env = CounselEnvironment()
    env.reset(difficulty="easy")
    first = env.witness.contradictions[0]
    env.step(CounselAction(tool="ask_question", text=f"{first.trigger_keywords[0]}?"))
    env.step(CounselAction(tool="present_evidence", exhibit_id=first.disprover_evidence_id))
    obs = env.step(CounselAction(tool="rest_case"))
    assert obs.reward > 0.0
    assert obs.reward_components["primary_reward"] == 1.0
    ok(f"Final reward {obs.reward:.3f} with components {obs.reward_components}")

    print("\n[TEST 4] Rubrics and Models")
    rubric = Total()
    rubric_reward = rubric(None, obs)
    assert rubric_reward > 0.0
    CounselAction(tool="rest_case")
    CounselObservation()
    CounselState()
    ok(f"Models instantiate and rubric score is {rubric_reward:.3f}")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print("Status: READY FOR LOCAL ROLLOUT VALIDATION")


if __name__ == "__main__":
    main()
