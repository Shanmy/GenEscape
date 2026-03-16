"""
Designer Agent for GenEscape.

Responsible for Stage 0: creating the initial scene description,
scene graph (YAML), and ground-truth solution for the escape room puzzle.
"""

import re
from typing import Optional

from genescape.llm_provider import BaseLLMProvider


_SYSTEM_PROMPT = """\
You are the Designer Agent for an escape room puzzle generation system.
Your role is to create rich, solvable escape room puzzles that are:
- Logically consistent (every required item is reachable)
- Appropriately challenging (no trivial shortcuts)
- Spatially coherent (objects are in sensible locations)
- Narratively engaging (the scene has atmosphere)

You MUST output your response using exactly these XML-like tags:
<description>...</description>
<scene_graph>...</scene_graph>
<solution>...</solution>

Do NOT include any text outside these tags.
"""

_DESIGN_TEMPLATE = """\
Design an escape room puzzle with the following parameters:

Scene keyword: {scene_keyword}
Available objects: {objects}
Solution length (number of steps): {solution_length}

Requirements:
1. Write a vivid one-paragraph scene description inside <description> tags.
   It should set the atmosphere and briefly describe what the player sees.

2. Create a scene graph in YAML inside <scene_graph> tags.
   The graph represents all objects in the room as a hierarchy:
   - Parent-child relationships represent spatial containment or attachment
     (e.g., a key inside a drawer, a note taped to a wall)
   - Each node must have:
       name: <string>        # unique object identifier
       state: <string>       # e.g., locked, unlocked, closed, open, on, off, hidden
       location: <string>    # brief spatial description (e.g., "north wall", "center of room")
       children: []          # list of child objects contained within/attached to this object
   - Include ALL objects from the provided list, plus any supporting objects needed
     (locks, codes, clues, containers) to make the solution work.
   - The room exit (door) must be locked initially.

3. Write a numbered ground-truth solution inside <solution> tags.
   Each step must be a concrete action: "Take [object] from [location]",
   "Use [object] on [object]", "Enter code [X] into [object]", "Open [object]", etc.
   The final step must be "Exit through the door" or equivalent.
   The solution must have EXACTLY {solution_length} steps.
   Every step must be logically necessary — no step can be skipped.

Example scene graph YAML structure:
```yaml
- name: room_door
  state: locked
  location: east wall
  children: []
- name: wooden_desk
  state: closed
  location: center of room
  children:
    - name: desk_drawer
      state: locked
      location: inside desk
      children:
        - name: brass_key
          state: available
          location: inside drawer
          children: []
```
"""


class DesignerAgent:
    """Creates the initial scene graph and ground-truth solution."""

    def __init__(self, provider: BaseLLMProvider) -> None:
        self.provider = provider

    def design(
        self,
        scene_keyword: str,
        objects: list[str],
        solution_length: int,
    ) -> dict:
        """
        Generate scene description, scene graph (YAML), and ground-truth solution.

        Returns:
            dict with keys: "description", "scene_graph" (YAML str), "solution" (list[str])
        """
        objects_str = ", ".join(objects)
        user_content = _DESIGN_TEMPLATE.format(
            scene_keyword=scene_keyword,
            objects=objects_str,
            solution_length=solution_length,
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response = self.provider.chat(messages)
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: str) -> dict:
        """Extract tagged sections from the LLM response."""
        description = _extract_tag(response, "description")
        scene_graph = _extract_tag(response, "scene_graph")
        solution_raw = _extract_tag(response, "solution")

        # Parse numbered solution steps
        solution_steps = _parse_numbered_list(solution_raw)

        return {
            "description": description.strip(),
            "scene_graph": scene_graph.strip(),
            "solution": solution_steps,
        }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _extract_tag(text: str, tag: str) -> str:
    """Extract content between <tag>...</tag>, handling optional whitespace."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    # Fallback: return everything if tag not found (graceful degradation)
    return text


def _parse_numbered_list(text: str) -> list[str]:
    """Parse a numbered list (1. step, 2. step, ...) into a Python list."""
    # Match lines like "1. ...", "1) ...", "Step 1: ..."
    lines = text.strip().splitlines()
    steps: list[str] = []
    current_step: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check if this line starts a new numbered step
        if re.match(r"^(\d+[\.\)]\s+|Step\s+\d+[:\.]\s*)", line, re.IGNORECASE):
            if current_step:
                steps.append(" ".join(current_step))
                current_step = []
            # Strip the number prefix
            step_text = re.sub(r"^(\d+[\.\)]\s+|Step\s+\d+[:\.]\s*)", "", line, flags=re.IGNORECASE)
            current_step.append(step_text.strip())
        else:
            # Continuation of previous step
            if current_step:
                current_step.append(line)
            elif line:
                current_step.append(line)

    if current_step:
        steps.append(" ".join(current_step))

    return steps if steps else [text.strip()]
