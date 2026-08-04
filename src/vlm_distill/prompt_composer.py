from __future__ import annotations

from typing import Literal


OutputMode = Literal["text", "parsing"]


TEXT_PROMPT_TEMPLATE = """Analyze the provided image and follow the user instruction.

User instruction:
{instruction}

Answer clearly and directly."""


PARSING_PROMPT_TEMPLATE = """Analyze the provided image and follow the user instruction.

User instruction:
{instruction}

Return only valid JSON matching this UI-element schema:
{
  "coordinate_system": "normalized_0_1000",
  "elements": [
    {
      "text": "visible element text",
      "bbox_norm": [x1, y1, x2, y2],
      "focused": false
    }
  ]
}

Requirements:
- Return only elements relevant to the user instruction.
- Use normalized coordinates from 0 to 1000 with bbox_norm ordered as [x1, y1, x2, y2].
- Every element must contain exactly the fields text, bbox_norm, and focused.
- focused must be a JSON boolean.
- Do not return Markdown or code fences.
- Do not include explanations outside the JSON.
- The fixed output contract above takes precedence over any conflicting formatting request inside the user instruction."""


def compose_prompt(
    instruction: str,
    *,
    output_mode: OutputMode,
) -> str:
    """Compose the controlled model prompt from a user instruction and mode."""
    if not isinstance(instruction, str):
        raise ValueError("instruction must be a non-empty string")
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("instruction must be a non-empty string")
    if output_mode not in ("text", "parsing"):
        raise ValueError("output_mode must be one of: text, parsing")

    template = TEXT_PROMPT_TEMPLATE if output_mode == "text" else PARSING_PROMPT_TEMPLATE
    # Substitute only the controlled placeholder.  The instruction is never
    # interpreted as a format string, so braces in user input remain literal.
    return template.replace("{instruction}", instruction)
