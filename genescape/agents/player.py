"""
Player Agent for GenEscape.

Simulates a human puzzle solver. Can work from:
- Text scene graph (Stage 1)
- 2D layout description (Stage 2)
- Actual image (Stage 3, uses vision)
"""

import re
from typing import Optional

from genescape.llm_provider import BaseLLMProvider


_SYSTEM_PROMPT = """\
You are the Player Agent simulating a human trying to escape a locked room.
You must reason carefully about what objects are available, their states, and
how they can be combined to eventually unlock the exit door.

Your output must be a numbered list of actions, one per line, in the format:
1. <action>
2. <action>
...

Each action should be specific and concrete, e.g.:
- "Examine the desk"
- "Take the brass key from inside the drawer"
- "Use the brass key on the padlock"
- "Open the cabinet door"
- "Enter code 4829 into the keypad"
- "Exit through the door"

Do NOT include any explanation or commentary — only the numbered action list.
"""

_GRAPH_SOLVE_TEMPLATE = """\
You are in the following escape room:

SCENE DESCRIPTION:
{description}

SCENE GRAPH (object hierarchy and states):
{scene_graph}

Based on the above information, determine the complete sequence of actions
needed to escape the room. Consider all object states, spatial relationships,
and logical dependencies between actions.

Provide the minimal sequence of steps to escape. Every step must be
logically necessary. Do not include redundant exploration steps unless
they are required to obtain a needed item.
"""

_LAYOUT_SOLVE_TEMPLATE = """\
You are in the following escape room:

SCENE DESCRIPTION:
{description}

2D LAYOUT (object positions and spatial arrangement):
{layout}

Based on the layout and scene description, determine the complete sequence of
actions needed to escape the room. Use the spatial arrangement to reason about
which objects are near each other and how they can be interacted with.

Provide the minimal sequence of steps to escape. Every step must be logically
necessary.
"""

_IMAGE_SOLVE_TEMPLATE = """\
You are looking at an image of an escape room. Study it carefully.

SCENE DESCRIPTION:
{description}

Based on what you can observe in the image (objects, their states, spatial
relationships, visible clues, locked/unlocked indicators), determine the
complete sequence of actions needed to escape the room.

Look for:
- Locked doors or containers
- Keys, codes, or tools
- Clues written on objects
- Objects that look interactive or moveable
- Any visual indicators of puzzle mechanisms

Provide the minimal sequence of steps to escape. Every step must be logically
necessary.
"""


class PlayerAgent:
    """Simulates a human solver attempting to escape the room."""

    def __init__(self, provider: BaseLLMProvider) -> None:
        self.provider = provider

    def solve_from_graph(self, scene_graph: str, description: str) -> list[str]:
        """
        Solve the puzzle using only the text scene graph.

        Args:
            scene_graph: YAML string of the scene graph
            description: Scene description text

        Returns:
            List of action step strings
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _GRAPH_SOLVE_TEMPLATE.format(
                    description=description,
                    scene_graph=scene_graph,
                ),
            },
        ]
        response = self.provider.chat(messages)
        return _parse_action_list(response)

    def solve_from_layout(self, layout: str, description: str) -> list[str]:
        """
        Solve the puzzle from a 2D layout description.

        Args:
            layout: YAML layout string with object positions
            description: Scene description text

        Returns:
            List of action step strings
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _LAYOUT_SOLVE_TEMPLATE.format(
                    description=description,
                    layout=layout,
                ),
            },
        ]
        response = self.provider.chat(messages)
        return _parse_action_list(response)

    def solve_from_image(self, image_path: str, description: str) -> list[str]:
        """
        Solve the puzzle by examining an image (uses vision model).

        Args:
            image_path: Path to the room image
            description: Scene description text

        Returns:
            List of action step strings
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _IMAGE_SOLVE_TEMPLATE.format(description=description),
            },
        ]
        response = self.provider.vision_chat(messages, image_path=image_path)
        return _parse_action_list(response)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _parse_action_list(text: str) -> list[str]:
    """Parse a numbered action list from LLM output."""
    lines = text.strip().splitlines()
    actions: list[str] = []
    current_action: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check if this line starts a new numbered step
        if re.match(r"^(\d+[\.\)]\s+)", line):
            if current_action:
                actions.append(" ".join(current_action))
                current_action = []
            step_text = re.sub(r"^(\d+[\.\)]\s+)", "", line)
            current_action.append(step_text.strip())
        else:
            if current_action:
                current_action.append(line)
            elif line and not line.startswith("#"):
                current_action.append(line)

    if current_action:
        actions.append(" ".join(current_action))

    return actions if actions else [text.strip()]
