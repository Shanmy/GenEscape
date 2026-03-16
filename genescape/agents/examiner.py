"""
Examiner Agent for GenEscape.

Responsible for:
- Comparing player solution vs ground-truth solution
- Identifying shortcuts, missing steps, wrong order, logical impossibilities
- Providing structured feedback for refinement
- Evaluating individual human player actions in interactive mode
"""

import re
from typing import Optional

from genescape.llm_provider import BaseLLMProvider


_CHECK_SYSTEM_PROMPT = """\
You are the Examiner Agent for an escape room puzzle generation system.
Your role is to rigorously compare a player's proposed solution against the
official ground-truth solution to identify any discrepancies.

You must check for:
1. Missing steps — actions in the official solution that the player skipped
2. Shortcut paths — player found a way to bypass intended puzzle mechanics
3. Wrong order — player performed actions in an illogical or impossible sequence
4. Logical impossibilities — player used an item before obtaining it, or
   interacted with a locked object without first unlocking it
5. Extra unnecessary steps — steps that are redundant and suggest the puzzle
   has ambiguity

If the player's solution is functionally equivalent to the official solution
(same logical steps, same result, minor wording differences allowed), output
exactly: SOLUTIONS_MATCH

Otherwise, output a bullet-point list of specific issues found, starting each
with "- ". Be precise about which step is problematic and why.

Do NOT output both SOLUTIONS_MATCH and issues — it must be one or the other.
"""

_CHECK_TEMPLATE = """\
OFFICIAL GROUND-TRUTH SOLUTION:
{official}

PLAYER'S PROPOSED SOLUTION:
{player}

Compare these solutions carefully and report any discrepancies.
"""

_REFINE_GRAPH_SYSTEM_PROMPT = """\
You are the Examiner Agent refining a scene graph based on identified puzzle issues.

Your task is to modify the YAML scene graph to fix the reported problems while:
- Preserving the overall puzzle theme and narrative
- Keeping the difficulty at the appropriate level
- Ensuring the official solution remains valid
- Blocking any identified shortcut paths
- Adding missing puzzle elements if steps cannot be completed

Output ONLY the corrected YAML scene graph with no additional explanation.
The output must be valid YAML starting with the first node definition.
"""

_REFINE_GRAPH_TEMPLATE = """\
CURRENT SCENE GRAPH:
{scene_graph}

IDENTIFIED ISSUES:
{feedback}

Fix the scene graph to address all the issues above. Ensure:
- Shortcuts are blocked (e.g., add locks, hide items more securely)
- All required items for each step are present and accessible only in the intended order
- Object states accurately reflect their initial condition

Output the corrected YAML scene graph:
"""

_EVALUATE_ACTION_SYSTEM_PROMPT = """\
You are an escape room game master evaluating a player's action.
Be encouraging but honest. If the action is correct, confirm it and hint at
what changed. If incorrect, give a gentle hint without revealing the answer.
Keep responses concise (1-3 sentences).
"""

_EVALUATE_ACTION_TEMPLATE = """\
ESCAPE ROOM DESCRIPTION:
{description}

OFFICIAL SOLUTION (for your reference only — do not reveal):
{official_solution}

CURRENT STEP (player should be on approximately step {current_step}):
Step {current_step}: {expected_step}

PLAYER'S ACTION:
"{action}"

Evaluate whether this action is correct, partially correct, or incorrect.
- If correct: confirm it worked and briefly describe the result.
- If partially correct: acknowledge what they tried and give a subtle hint.
- If incorrect: gently indicate it didn't work and give a small hint.
Do not reveal the exact next step or future steps.
"""


class ExaminerAgent:
    """Compares solutions, provides feedback, and evaluates human player actions."""

    def __init__(self, provider: BaseLLMProvider) -> None:
        self.provider = provider

    def check(
        self,
        official_solution: list[str],
        player_solution: list[str],
    ) -> list[str]:
        """
        Compare player solution vs official solution.

        Returns:
            Empty list if solutions match; otherwise a list of feedback bullet points.
        """
        official_str = _format_solution(official_solution)
        player_str = _format_solution(player_solution)

        messages = [
            {"role": "system", "content": _CHECK_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _CHECK_TEMPLATE.format(
                    official=official_str,
                    player=player_str,
                ),
            },
        ]

        response = self.provider.chat(messages).strip()

        if "SOLUTIONS_MATCH" in response.upper():
            return []

        # Parse bullet points
        feedback = _parse_bullet_points(response)
        return feedback if feedback else [response]

    def refine_graph(self, scene_graph: str, feedback: list[str]) -> str:
        """
        Refine the scene graph based on examiner feedback.

        Args:
            scene_graph: Current YAML scene graph string
            feedback: List of feedback bullet points

        Returns:
            Refined YAML scene graph string
        """
        feedback_str = "\n".join(f"- {item}" for item in feedback)

        messages = [
            {"role": "system", "content": _REFINE_GRAPH_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _REFINE_GRAPH_TEMPLATE.format(
                    scene_graph=scene_graph,
                    feedback=feedback_str,
                ),
            },
        ]

        response = self.provider.chat(messages).strip()
        # Strip any markdown code fence if present
        return _strip_code_fence(response)

    def evaluate_action(
        self,
        action: str,
        official_solution: list[str],
        current_step: int,
        description: str,
    ) -> str:
        """
        Evaluate a human player's action during interactive play.

        Args:
            action: The action string typed by the human player
            official_solution: The full official solution list
            current_step: The step index the player is currently on (0-based)
            description: Scene description for context

        Returns:
            Guidance string for the player
        """
        step_idx = min(current_step, len(official_solution) - 1)
        expected_step = official_solution[step_idx] if official_solution else "Exit through the door"

        official_str = _format_solution(official_solution)

        messages = [
            {"role": "system", "content": _EVALUATE_ACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _EVALUATE_ACTION_TEMPLATE.format(
                    description=description,
                    official_solution=official_str,
                    current_step=current_step + 1,
                    expected_step=expected_step,
                    action=action,
                ),
            },
        ]

        return self.provider.chat(messages).strip()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _format_solution(steps: list[str]) -> str:
    """Format a list of steps as a numbered string."""
    return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))


def _parse_bullet_points(text: str) -> list[str]:
    """Extract bullet-point items from text."""
    items: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("- ") or line.startswith("* ") or line.startswith("• "):
            if current:
                items.append(" ".join(current))
                current = []
            current.append(line[2:].strip())
        elif current:
            # Continuation line
            current.append(line)
        elif line:
            # Non-bullet text — treat as a single feedback item
            items.append(line)

    if current:
        items.append(" ".join(current))

    return items


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences (```yaml ... ```) from text."""
    # Remove ```yaml or ``` at the start
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    # Remove ``` at the end
    text = re.sub(r"\n?```$", "", text.strip())
    return text.strip()
