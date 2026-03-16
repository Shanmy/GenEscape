"""
Human Interface for GenEscape.

Allows a human player to interactively solve a generated escape room puzzle.
The ExaminerAgent provides real-time feedback on each action without
revealing the ground-truth solution.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from genescape.llm_provider import BaseLLMProvider
from genescape.agents.examiner import ExaminerAgent


_WELCOME_BANNER = """
╔══════════════════════════════════════════════════════════╗
║           WELCOME TO GENESCAPE — INTERACTIVE MODE        ║
╚══════════════════════════════════════════════════════════╝
"""

_HELP_TEXT = """
Available commands:
  <any action>   — Type what you want to do (e.g., "Take the key from the drawer")
  hint           — Request a hint for your current situation
  look           — Describe what you see in the room
  inventory      — List items you've picked up (tracked by you)
  quit / exit    — Give up and exit
  solution       — Reveal the full solution (gives up the game)
"""

_HINT_PROMPT_TEMPLATE = """\
The player is stuck and needs a hint.

SCENE DESCRIPTION:
{description}

OFFICIAL SOLUTION (do not reveal directly):
{official_solution}

PLAYER'S CURRENT PROGRESS (they have completed approximately {current_step} steps):
{completed_steps}

Give a helpful hint for what the player should do next.
Be encouraging but do not give away the exact action — just point them in the right direction.
Keep the hint to 1-2 sentences.
"""

_LOOK_PROMPT_TEMPLATE = """\
The player has asked to look around the room.

SCENE DESCRIPTION:
{description}

PLAYER'S CURRENT PROGRESS ({current_step} steps completed):
{completed_steps}

Describe what the player currently sees in the room in 3-5 sentences.
Only describe objects and states that make sense given their current progress.
Do not reveal puzzle solutions — just describe the visible environment.
Be atmospheric and immersive.
"""

_ESCAPE_KEYWORDS = [
    "exit through the door",
    "escape through the door",
    "open the door and leave",
    "leave the room",
    "escaped",
    "freedom",
    "exit the room",
    "walk out",
    "go through the door",
    "step outside",
]


class HumanInterface:
    """Interactive puzzle-playing interface for human players."""

    def __init__(self, provider: BaseLLMProvider) -> None:
        self.provider = provider
        self.examiner = ExaminerAgent(provider)

    def play(
        self,
        image_path: str,
        description: str,
        official_solution: list[str],
    ) -> None:
        """
        Launch an interactive puzzle session.

        Args:
            image_path: Path to the room image to display
            description: Scene description text
            official_solution: The ground-truth solution (used by examiner for feedback)
        """
        print(_WELCOME_BANNER)

        # Display image info
        image_file = Path(image_path)
        if image_file.exists() and not str(image_path).endswith("_description.txt"):
            print(f"  Room image: {image_path}")
            print(f"  Open the image to see your escape room!")
            # Try to open the image with the system viewer
            self._try_open_image(image_path)
        else:
            print(f"  [Note] No image available at: {image_path}")
            if str(image_path).endswith("_description.txt"):
                print(f"  A scene description was saved instead (image generation unavailable).")

        print()
        print("  SCENE:")
        print(f"  {description}")
        print()
        print(f"  Your goal: Find a way to escape this room!")
        print(f"  The solution has {len(official_solution)} steps.")
        print()
        print(_HELP_TEXT)
        print("─" * 60)

        current_step: int = 0
        completed_actions: list[str] = []
        inventory: list[str] = []
        escaped = False

        while not escaped:
            try:
                raw_input = input("\n> What do you do? ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  Interrupted. Thanks for playing!")
                break

            if not raw_input:
                continue

            action_lower = raw_input.lower()

            # --- Special commands ---
            if action_lower in ("quit", "exit", "give up"):
                print("\n  You decide to give up. Better luck next time!")
                print(f"\n  The solution was:")
                for i, step in enumerate(official_solution, 1):
                    print(f"    {i}. {step}")
                break

            if action_lower == "solution":
                confirm = input("  Reveal the full solution? (yes/no): ").strip().lower()
                if confirm in ("yes", "y"):
                    print("\n  Full solution:")
                    for i, step in enumerate(official_solution, 1):
                        print(f"    {i}. {step}")
                    print()
                    escaped_confirm = input("  Do you want to continue playing? (yes/no): ").strip().lower()
                    if escaped_confirm not in ("yes", "y"):
                        break
                continue

            if action_lower == "help":
                print(_HELP_TEXT)
                continue

            if action_lower == "inventory":
                if inventory:
                    print(f"\n  You are carrying: {', '.join(inventory)}")
                else:
                    print("\n  You are not carrying anything yet.")
                continue

            if action_lower == "look":
                look_text = self._generate_look(
                    description, official_solution, current_step, completed_actions
                )
                print(f"\n  {look_text}")
                continue

            if action_lower == "hint":
                hint_text = self._generate_hint(
                    description, official_solution, current_step, completed_actions
                )
                print(f"\n  Hint: {hint_text}")
                continue

            # --- Regular action ---
            print()
            feedback = self.examiner.evaluate_action(
                action=raw_input,
                official_solution=official_solution,
                current_step=current_step,
                description=description,
            )
            print(f"  {feedback}")

            # Check if this action advances progress
            if self._is_correct_action(raw_input, official_solution, current_step):
                completed_actions.append(raw_input)
                # Track items picked up
                self._update_inventory(raw_input, inventory)
                current_step += 1
                print(f"\n  [Progress: {current_step}/{len(official_solution)} steps completed]")

                # Check escape condition
                if self._check_escaped(raw_input, official_solution, current_step):
                    escaped = True
                    self._celebrate_escape(completed_actions, current_step)

            elif self._check_escaped(raw_input, official_solution, len(official_solution)):
                # Direct escape detection from action text
                escaped = True
                current_step = len(official_solution)
                self._celebrate_escape(completed_actions + [raw_input], current_step)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_correct_action(
        self,
        action: str,
        solution: list[str],
        current_step: int,
    ) -> bool:
        """
        Heuristic check: does this action correspond to the current solution step?
        Uses keyword overlap between the action and expected step.
        """
        if current_step >= len(solution):
            return self._check_escaped(action, solution, current_step)

        expected = solution[current_step].lower()
        action_lower = action.lower()

        # Check for significant keyword overlap
        expected_words = set(w for w in expected.split() if len(w) > 3)
        action_words = set(w for w in action_lower.split() if len(w) > 3)
        overlap = expected_words & action_words

        # Heuristic: if more than 40% of key words overlap, consider it correct
        if expected_words and len(overlap) / len(expected_words) >= 0.4:
            return True

        return False

    def _check_escaped(
        self,
        action: str,
        solution: list[str],
        current_step: int,
    ) -> bool:
        """Check if the action represents escaping the room."""
        action_lower = action.lower()

        # Check escape keywords
        if any(kw in action_lower for kw in _ESCAPE_KEYWORDS):
            return True

        # Check if we're on the last step and the action roughly matches
        if solution and current_step >= len(solution) - 1:
            last_step = solution[-1].lower()
            if any(kw in last_step for kw in _ESCAPE_KEYWORDS):
                if any(kw in action_lower for kw in ["exit", "door", "escape", "leave", "out"]):
                    return True

        return False

    def _update_inventory(self, action: str, inventory: list[str]) -> None:
        """Update inventory based on 'take'/'pick up'/'grab' actions."""
        action_lower = action.lower()
        take_patterns = ["take ", "pick up ", "grab ", "collect ", "get "]
        for pattern in take_patterns:
            if pattern in action_lower:
                idx = action_lower.index(pattern) + len(pattern)
                # Extract the item name (stop at prepositions)
                remainder = action[idx:]
                for stop_word in [" from ", " off ", " out of ", " in "]:
                    if stop_word in remainder.lower():
                        remainder = remainder[: remainder.lower().index(stop_word)]
                item = remainder.strip().rstrip(".,!?")
                if item and item not in inventory:
                    inventory.append(item)
                break

    def _generate_hint(
        self,
        description: str,
        solution: list[str],
        current_step: int,
        completed: list[str],
    ) -> str:
        """Generate a contextual hint using the LLM."""
        official_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(solution))
        completed_str = (
            "\n".join(f"- {a}" for a in completed) if completed else "(none yet)"
        )

        messages = [
            {
                "role": "user",
                "content": _HINT_PROMPT_TEMPLATE.format(
                    description=description,
                    official_solution=official_str,
                    current_step=current_step,
                    completed_steps=completed_str,
                ),
            }
        ]
        return self.provider.chat(messages).strip()

    def _generate_look(
        self,
        description: str,
        solution: list[str],
        current_step: int,
        completed: list[str],
    ) -> str:
        """Generate a room description for the current game state."""
        official_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(solution))
        completed_str = (
            "\n".join(f"- {a}" for a in completed) if completed else "(none yet)"
        )

        messages = [
            {
                "role": "user",
                "content": _LOOK_PROMPT_TEMPLATE.format(
                    description=description,
                    current_step=current_step,
                    completed_steps=completed_str,
                ),
            }
        ]
        return self.provider.chat(messages).strip()

    @staticmethod
    def _celebrate_escape(completed_actions: list[str], steps: int) -> None:
        """Print a congratulations message."""
        print("\n" + "=" * 60)
        print("  CONGRATULATIONS! YOU ESCAPED!")
        print("=" * 60)
        print(f"\n  You solved the puzzle in {steps} actions!")
        if completed_actions:
            print("\n  Your solution:")
            for i, action in enumerate(completed_actions, 1):
                print(f"    {i}. {action}")
        print()

    @staticmethod
    def _try_open_image(image_path: str) -> None:
        """Attempt to open the image with the system's default viewer."""
        import subprocess
        import sys

        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", image_path])
            elif sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", image_path])
            elif sys.platform == "win32":
                os.startfile(image_path)
        except Exception:
            pass  # Silently ignore — image display is optional
