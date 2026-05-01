"""voting_access — procedural civic facts about voting in the US."""

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.solver import generate

from p3.lib import load_samples
from p3.scorers import appropriate_refusal, ground_truth_match, rubric_judge

HERE = Path(__file__).parent


@task
def voting_access() -> Task:
    return Task(
        dataset=load_samples(HERE / "tasks.jsonl"),
        solver=generate(),
        scorer=[
            # Regex mode + word-boundary patterns in tasks.jsonl. Substring
            # mode false-positives on short targets ("no" inside "knot was
            # not the issue", "yes" inside "yesterday").
            ground_truth_match(mode="regex"),
            rubric_judge(),
            appropriate_refusal(),
        ],
    )
