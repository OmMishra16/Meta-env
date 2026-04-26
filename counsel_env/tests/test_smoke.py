import random

from counsel_env.models import CounselAction
from counsel_env.evaluation import evaluate_agent, present_all_agent, keyword_spam_agent, oracle_scripted_agent
from counsel_env.server.app import app
from counsel_env.server.case_generator import generate_case
from counsel_env.server.counsel_env_environment import CounselEnvironment
from counsel_env.server.witness import Contradiction, Witness


def _trigger_and_surface_first(env: CounselEnvironment):
    contradiction = env.witness.contradictions[0]
    env.step(CounselAction(tool="ask_question", text=f"{contradiction.trigger_keywords[0]}?"))
    env.step(CounselAction(tool="present_evidence", exhibit_id=contradiction.disprover_evidence_id))
    return contradiction


class TestWitnessLogic:
    def test_witness_trigger(self):
        contradiction = Contradiction(
            cid="c1",
            trigger_keywords=["where were you", "alibi"],
            sealed_claim="I was at the diner all night.",
            disprover_evidence_id="phone_log",
        )
        witness = Witness({"motive": "No motive."}, [contradiction])

        response = witness.respond_to_question("Where were you that night?")

        assert response == contradiction.sealed_claim
        assert contradiction.triggered is True
        assert witness.committed_claims == [contradiction.sealed_claim]

    def test_evidence_surface(self):
        contradiction = Contradiction(
            cid="c1",
            trigger_keywords=["alibi"],
            sealed_claim="I was at the diner.",
            disprover_evidence_id="phone_log",
        )
        witness = Witness({}, [contradiction])

        witness.respond_to_question("Tell us your alibi.")
        response = witness.react_to_evidence("phone_log")

        assert "stammers" in response.lower()
        assert contradiction.surfaced is True

    def test_no_false_positive(self):
        contradiction = Contradiction(
            cid="c1",
            trigger_keywords=["alibi"],
            sealed_claim="I was at the diner.",
            disprover_evidence_id="phone_log",
        )
        witness = Witness({}, [contradiction])

        response = witness.react_to_evidence("phone_log")

        assert "no comment" in response.lower()
        assert contradiction.surfaced is False


class TestCaseGeneration:
    def test_curriculum_difficulty_shapes_cases(self):
        easy = generate_case(difficulty="easy")
        medium = generate_case(difficulty="medium")
        hard = generate_case(difficulty="hard")

        assert len(easy["contradictions"]) == 1
        assert len(medium["contradictions"]) == 2
        assert len(hard["contradictions"]) >= 3
        assert "irrelevant_weather_report" in hard["evidence"]

    def test_curriculum_sampling_distribution(self):
        cases = [
            generate_case(curriculum_stage="mixed", distribution={"easy": 1.0})
            for _ in range(5)
        ]

        assert {case["difficulty"] for case in cases} == {"easy"}

    def test_seed_replay_is_deterministic(self):
        env_a = CounselEnvironment()
        env_b = CounselEnvironment()
        obs_a = env_a.reset(seed=1234, difficulty="hard")
        obs_b = env_b.reset(seed=1234, difficulty="hard")

        assert obs_a.case_id == obs_b.case_id
        assert obs_a.case_brief == obs_b.case_brief
        assert obs_a.evidence_descriptions == obs_b.evidence_descriptions
        assert [c.cid for c in env_a.witness.contradictions] == [c.cid for c in env_b.witness.contradictions]


class TestEnvironmentLogic:
    def test_space_demo_routes_registered(self):
        paths = {route.path for route in app.routes}

        assert "/" in paths
        assert "/demo" in paths
        assert "/demo/api/reset" in paths
        assert "/demo/api/step" in paths

    def test_reset_state(self):
        env = CounselEnvironment()
        obs = env.reset(difficulty="easy")

        assert obs.questions_remaining == 15
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.case_brief
        assert obs.available_evidence
        assert obs.evidence_descriptions
        assert set(obs.available_evidence) == set(obs.evidence_descriptions)
        assert obs.case_id == env.case["case_id"]
        assert obs.difficulty == "easy"
        assert env.state.contradictions_total == 1
        assert env.state.questions_used == 0

    def test_question_budget(self):
        env = CounselEnvironment()
        env.reset(difficulty="easy")

        obs = None
        for i in range(env.QUESTION_BUDGET):
            obs = env.step(CounselAction(tool="ask_question", text=f"Off-topic question {i}?"))

        assert obs is not None
        assert obs.done is True
        assert obs.questions_remaining == 0

        after_done = env.step(CounselAction(tool="ask_question", text="One too many?"))
        assert after_done.done is True
        assert "already over" in after_done.witness_response.lower()

    def test_duplicate_question_penalty(self):
        clean_env = CounselEnvironment()
        clean_env.reset(difficulty="easy")
        _trigger_and_surface_first(clean_env)
        clean_reward = clean_env.step(CounselAction(tool="rest_case")).reward

        duplicate_env = CounselEnvironment()
        duplicate_env.reset(difficulty="easy")
        contradiction = duplicate_env.witness.contradictions[0]
        question = f"{contradiction.trigger_keywords[0]}?"
        duplicate_env.step(CounselAction(tool="ask_question", text=question))
        duplicate_response = duplicate_env.step(CounselAction(tool="ask_question", text=question))
        duplicate_env.step(CounselAction(tool="present_evidence", exhibit_id=contradiction.disprover_evidence_id))
        duplicate_reward = duplicate_env.step(CounselAction(tool="rest_case")).reward

        assert "already asked" in duplicate_response.witness_response.lower()
        assert duplicate_env.duplicate_question_count == 1
        assert duplicate_reward < clean_reward

    def test_reward_components_primary_dominates(self):
        env = CounselEnvironment()
        env.reset(difficulty="easy")
        _trigger_and_surface_first(env)
        obs = env.step(CounselAction(tool="rest_case"))

        assert obs.reward_components["primary_reward"] == 1.0
        assert obs.reward >= 0.8
        assert obs.reward_components["auxiliary_reward_raw"] > 0.0
        assert obs.reward_components["evidence_timing_successes"] >= 1.0

    def test_no_infinite_evidence_loop(self):
        env = CounselEnvironment()
        env.reset(difficulty="easy")
        exhibit = env.available_evidence[0] if hasattr(env, "available_evidence") else env.case["evidence"].keys()
        exhibit_id = next(iter(env.case["evidence"].keys()))

        for _ in range(env.MAX_ACTIONS + 1):
            obs = env.step(CounselAction(tool="present_evidence", exhibit_id=exhibit_id))
            if obs.done:
                break

        assert obs.done is True
        assert env.action_count >= env.MAX_ACTIONS

    def test_transcript_export_contains_labels(self):
        env = CounselEnvironment()
        env.reset(seed=55, difficulty="easy", episode_id="demo")
        _trigger_and_surface_first(env)
        obs = env.step(CounselAction(tool="rest_case"))

        payload = env.export_transcript_json()
        markdown = env.export_transcript_markdown()

        assert payload["episode_id"] == "demo"
        assert payload["seed"] == 55
        assert payload["events"]
        assert payload["reward_components"]["total_reward"] == obs.reward
        assert "Triggered:" in markdown
        assert "Surfaced:" in markdown

    def test_state_tracks_extended_diagnostics(self):
        env = CounselEnvironment()
        env.reset(difficulty="easy")
        _trigger_and_surface_first(env)
        state = env.state

        assert state.contradictions_triggered == 1
        assert state.contradictions_surfaced == 1
        assert state.evidence_timing_successes == 1
        assert state.action_count >= 2


class TestRolloutValidation:
    def test_scripted_agent_success(self):
        successes = 0

        for _ in range(10):
            env = CounselEnvironment()
            env.reset(curriculum_stage="mixed")
            _trigger_and_surface_first(env)
            obs = env.step(CounselAction(tool="rest_case"))
            if obs.reward > 0:
                successes += 1

        assert successes >= 1

    def test_random_agent_baseline(self):
        rng = random.Random(7)
        rewards = []

        for _ in range(10):
            env = CounselEnvironment()
            obs = env.reset(difficulty="hard")
            for _ in range(8):
                if rng.random() < 0.7:
                    obs = env.step(
                        CounselAction(tool="ask_question", text=f"Random question {rng.randint(1, 999)}?")
                    )
                else:
                    obs = env.step(
                        CounselAction(tool="present_evidence", exhibit_id=rng.choice(obs.available_evidence))
                    )
                if obs.done:
                    break
            rewards.append(env.step(CounselAction(tool="rest_case")).reward)

        avg_reward = sum(rewards) / len(rewards)
        assert avg_reward < 0.35

    def test_reward_hacking_baselines_stay_below_oracle(self):
        seeds = [9001, 9002, 9003, 9004, 9005]
        spam_rows, _ = evaluate_agent("keyword_spam", keyword_spam_agent, seeds, transcript_limit=0)
        present_rows, _ = evaluate_agent("present_all", present_all_agent, seeds, transcript_limit=0)
        oracle_rows, _ = evaluate_agent("scripted_oracle", oracle_scripted_agent, seeds, transcript_limit=0)

        spam_reward = sum(row["reward"] for row in spam_rows) / len(spam_rows)
        present_reward = sum(row["reward"] for row in present_rows) / len(present_rows)
        oracle_reward = sum(row["reward"] for row in oracle_rows) / len(oracle_rows)

        assert oracle_reward > spam_reward
        assert oracle_reward > present_reward
        assert present_reward == 0.0


class TestEdgeCases:
    def test_empty_question(self):
        env = CounselEnvironment()
        env.reset(difficulty="easy")

        obs = env.step(CounselAction(tool="ask_question", text=""))

        assert "proper question" in obs.witness_response.lower()
        assert env.inadmissible_count == 1

    def test_invalid_exhibit(self):
        env = CounselEnvironment()
        env.reset(difficulty="easy")

        obs = env.step(CounselAction(tool="present_evidence", exhibit_id="nonexistent_exhibit"))

        assert "invalid exhibit" in obs.witness_response.lower()
        assert env.inadmissible_count == 1

    def test_extremely_long_question(self):
        env = CounselEnvironment()
        env.reset(difficulty="easy")
        long_question = "What is your alibi " + ("please explain " * 200)

        obs = env.step(CounselAction(tool="ask_question", text=long_question))

        assert obs.witness_response
        assert obs.questions_remaining == env.QUESTION_BUDGET - 1

    def test_no_trigger_keywords_case(self):
        witness = Witness(
            {"background": "I worked late."},
            [
                Contradiction(
                    cid="c1",
                    trigger_keywords=[],
                    sealed_claim="Hidden claim.",
                    disprover_evidence_id="log",
                )
            ],
        )

        response = witness.respond_to_question("What is your favorite color?")
        evidence_response = witness.react_to_evidence("log")

        assert response == "I don't recall."
        assert "no comment" in evidence_response.lower()
