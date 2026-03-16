"""
Builder Agent for GenEscape.

Responsible for:
- Creating 2D layout from scene graph (Stage 2)
- Generating photorealistic room images (Stage 3)
- Refining layout and images based on Examiner feedback
"""

import re
from pathlib import Path
from typing import Optional

from genescape.llm_provider import BaseLLMProvider


_LAYOUT_SYSTEM_PROMPT = """\
You are the Builder Agent for an escape room puzzle generation system.
Your role is to translate a scene graph into a precise 2D spatial layout.

Output ONLY valid YAML — no explanation, no markdown fences.
The YAML must be a list of objects, each with these required fields:
  - name: <string>          # object identifier (match scene graph name)
  - label: <string>         # human-readable display name
  - x: <float>              # horizontal position, 0.0 (left) to 1.0 (right)
  - y: <float>              # vertical position, 0.0 (top) to 1.0 (bottom)
  - width: <float>          # relative width, 0.0 to 1.0
  - height: <float>         # relative height, 0.0 to 1.0
  - state: <string>         # current state from scene graph
  - layer: <int>            # rendering layer (1=floor/walls, 2=furniture, 3=small objects)

Objects must not overlap unless one is logically on top of/inside another (use layer).
The room door should be on one of the walls (x=0 or x=1 or y=0 or y=1).
Distribute objects naturally across the room — avoid clustering everything in one corner.
"""

_LAYOUT_TEMPLATE = """\
Convert the following scene graph into a 2D room layout.

SCENE DESCRIPTION:
{description}

SCENE GRAPH:
{scene_graph}

Rules:
- Room dimensions are normalized to 1.0 x 1.0
- Wall objects (doors, windows, shelves mounted on walls) should be near the edges (x<0.15, x>0.85, y<0.15, y>0.85)
- Floor objects (desks, chairs, tables) should be in the interior
- Small objects (keys, notes, books) should have width and height < 0.08
- Large furniture (desks, cabinets, bookshelves) should have width 0.15-0.35
- Locked objects should have their 'state' field reflect that (e.g., state: locked)

Output the YAML layout:
"""

_REFINE_LAYOUT_SYSTEM_PROMPT = """\
You are the Builder Agent refining a 2D room layout based on puzzle issues.

Output ONLY valid YAML — no explanation, no markdown fences.
Maintain the same structure (list of objects with name, label, x, y, width, height, state, layer).
"""

_REFINE_LAYOUT_TEMPLATE = """\
CURRENT LAYOUT:
{layout}

SCENE GRAPH (for reference):
{scene_graph}

IDENTIFIED ISSUES:
{feedback}

Adjust the layout to fix these issues:
- If an object is too accessible/visible when it should be hidden, move it
  to be a child of a container or place it in a less prominent position
- If objects that should interact are too far apart, bring them closer
- Ensure puzzle flow is reflected in the spatial arrangement

Output the corrected YAML layout:
"""

_IMAGE_PROMPT_TEMPLATE = """\
Create a photorealistic escape room image with the following specifications:

ROOM TYPE AND ATMOSPHERE:
{description}

VISUAL STYLE:
- Photorealistic rendering, high detail, cinematic quality
- Warm, slightly dim atmospheric lighting (overhead light + practical lamps)
- Single-point perspective from slightly above eye level, showing the full room
- Slightly dusty, lived-in feel with visible texture on surfaces
- The room should feel immersive and three-dimensional

ROOM DIMENSIONS:
- Roughly 5m x 5m room with ~2.5m ceiling height
- Visible floor (hardwood or tile), walls (plaster or brick), ceiling

OBJECT PLACEMENT (from 2D layout, normalized 0.0-1.0 coordinates):
{object_list}

PUZZLE ELEMENT VISIBILITY:
- All interactive objects must be clearly visible and identifiable
- Locked objects must visually show their locked state (padlock, keyhole, chains)
- Keys, codes, and clues must be visible but require attention to spot
- Object states must be clear from visual appearance alone
- Any written clues or codes on objects must be legible

IMPORTANT: This is a puzzle room — every visible object serves a purpose.
The image must contain ALL listed objects in their specified positions.
No people or characters should be present in the room.
"""

_REFINE_IMAGE_SYSTEM_PROMPT = """\
You are analyzing an escape room image to understand what needs to be changed.
Describe specifically what objects are visible, their positions, states, and
what is wrong based on the feedback provided.
Then write a complete, detailed image generation prompt for the corrected version.
"""

_REFINE_IMAGE_TEMPLATE = """\
I have an escape room image that needs to be corrected.

CURRENT SCENE DESCRIPTION:
{description}

CURRENT LAYOUT:
{layout}

ISSUES TO FIX:
{feedback}

Please analyze the current image (attached) and then write a complete,
detailed image generation prompt that would produce a corrected version
of the room addressing all the listed issues.

The new prompt should describe the full room (not just the changes),
maintaining the same atmosphere and theme but fixing the identified problems.

Output only the image generation prompt text, starting with "Create a photorealistic..."
"""


class BuilderAgent:
    """Creates 2D layouts and generates/refines room images."""

    def __init__(self, provider: BaseLLMProvider) -> None:
        self.provider = provider

    def create_layout(self, scene_graph: str, description: str) -> str:
        """
        Convert a scene graph into a 2D spatial layout (YAML).

        Args:
            scene_graph: YAML scene graph string
            description: Scene description text

        Returns:
            YAML layout string with object positions
        """
        messages = [
            {"role": "system", "content": _LAYOUT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _LAYOUT_TEMPLATE.format(
                    description=description,
                    scene_graph=scene_graph,
                ),
            },
        ]

        response = self.provider.chat(messages).strip()
        return _strip_code_fence(response)

    def refine_layout(self, layout: str, feedback: list[str], scene_graph: str) -> str:
        """
        Refine an existing layout based on examiner feedback.

        Args:
            layout: Current YAML layout string
            feedback: List of feedback bullet points
            scene_graph: YAML scene graph for reference

        Returns:
            Refined YAML layout string
        """
        feedback_str = "\n".join(f"- {item}" for item in feedback)

        messages = [
            {"role": "system", "content": _REFINE_LAYOUT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _REFINE_LAYOUT_TEMPLATE.format(
                    layout=layout,
                    scene_graph=scene_graph,
                    feedback=feedback_str,
                ),
            },
        ]

        response = self.provider.chat(messages).strip()
        return _strip_code_fence(response)

    def generate_image(
        self,
        layout: str,
        description: str,
        scene_graph: str,
        output_path: str,
    ) -> str:
        """
        Build a detailed image generation prompt and generate the room image.

        Args:
            layout: YAML layout with object positions
            description: Scene description text
            scene_graph: YAML scene graph for additional context
            output_path: File path to save the generated image

        Returns:
            Path to the saved image (or description file if generation failed)
        """
        prompt = self._build_image_prompt(layout, description, scene_graph)
        return self.provider.generate_image(prompt, output_path)

    def refine_image(
        self,
        image_path: str,
        layout: str,
        feedback: list[str],
        description: str,
        output_path: str,
    ) -> str:
        """
        Analyze the current image + feedback, generate a corrected image.

        Args:
            image_path: Path to the current room image
            layout: Current YAML layout string
            feedback: List of feedback bullet points from examiner
            description: Scene description text
            output_path: File path to save the refined image

        Returns:
            Path to the saved refined image
        """
        feedback_str = "\n".join(f"- {item}" for item in feedback)

        messages = [
            {"role": "system", "content": _REFINE_IMAGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _REFINE_IMAGE_TEMPLATE.format(
                    description=description,
                    layout=layout,
                    feedback=feedback_str,
                ),
            },
        ]

        # Use vision if the image exists and is a real image (not a fallback description)
        path_lower = image_path.lower()
        is_real_image = (
            Path(image_path).exists()
            and not path_lower.endswith("_description.txt")
        )

        if is_real_image:
            new_prompt = self.provider.vision_chat(messages, image_path=image_path)
        else:
            new_prompt = self.provider.chat(messages)

        new_prompt = new_prompt.strip()
        # If the model returned extra text before the prompt, extract just the prompt
        if "Create a photorealistic" in new_prompt:
            idx = new_prompt.index("Create a photorealistic")
            new_prompt = new_prompt[idx:]

        return self.provider.generate_image(new_prompt, output_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_image_prompt(layout: str, description: str, scene_graph: str) -> str:
        """Build a detailed, structured image generation prompt."""
        import yaml as _yaml

        # Parse layout YAML to extract object list
        object_lines: list[str] = []
        try:
            layout_data = _yaml.safe_load(layout)
            if isinstance(layout_data, list):
                for obj in layout_data:
                    if isinstance(obj, dict):
                        name = obj.get("label") or obj.get("name", "unknown")
                        x = obj.get("x", 0.5)
                        y = obj.get("y", 0.5)
                        state = obj.get("state", "")
                        # Convert normalized coords to rough position descriptions
                        h_pos = "left" if x < 0.35 else ("right" if x > 0.65 else "center")
                        v_pos = "top (far wall)" if y < 0.35 else ("bottom (near)" if y > 0.65 else "middle")
                        state_str = f" [{state}]" if state else ""
                        object_lines.append(
                            f"  - {name}{state_str}: positioned at {h_pos}-{v_pos} of room"
                        )
        except Exception:
            # Fallback: use raw layout text
            object_lines = ["  " + line for line in layout.splitlines() if line.strip()]

        object_list = "\n".join(object_lines) if object_lines else "  " + layout

        return _IMAGE_PROMPT_TEMPLATE.format(
            description=description,
            object_list=object_list,
        )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences (```yaml ... ```) from text."""
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    return text.strip()
