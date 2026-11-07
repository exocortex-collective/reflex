"""Natural language command parser for Reflex."""

import json
from dataclasses import dataclass
from typing import Any, Literal, Optional

import httpx
from cortex_utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedCommand:
    """Structured representation of a parsed command."""

    action: Literal["move", "tag", "archive", "show"]
    target: str  # that, this, the last, recent
    target_keywords: list[str]  # Keywords to help resolve target
    parameters: dict[str, Any]  # Action-specific parameters


class CommandParser:
    """Parse natural language commands using LLM."""

    PARSE_PROMPT = """You are a command parser for a personal knowledge capture system.

Parse this natural language command into structured JSON:

Command: "{command}"

Available actions:
- **move**: Change entry category (person, project, idea, admin, inbox)
- **tag**: Add tags to entry
- **archive**: Archive entry (set status=archived)
- **show**: Display entry details

Target references:
- "that" / "this" / "the last one" → most recent entry
- "that [keyword]" → entry matching keyword

Return JSON with this structure:
{{
  "action": "move|tag|archive|show",
  "target": "that|this|last",
  "target_keywords": ["keyword1", "keyword2"],
  "parameters": {{
    "new_category": "project",
    "tags": ["urgent", "followup"],
    "reason": "completed"
  }}
}}

Examples:
- "move that idea about Foo into my projects" → {{"action": "move", "target": "that", "target_keywords": ["idea", "Foo"], "parameters": {{"new_category": "project"}}}}
- "tag the last note with 'urgent'" → {{"action": "tag", "target": "last", "target_keywords": ["note"], "parameters": {{"tags": ["urgent"]}}}}
- "archive that inbox item" → {{"action": "archive", "target": "that", "target_keywords": ["inbox"], "parameters": {{}}}}

Return ONLY valid JSON, no other text."""

    def __init__(self, litellm_url: str, model: str, api_key: Optional[str] = None):
        """Initialize command parser.

        Args:
            litellm_url: LiteLLM proxy URL
            model: Model to use for parsing. The bot passes the TIER 1 model,
                which is local -- see bot.py for why the tier 2 model must not
                be used here.
            api_key: Sent as "Authorization: Bearer <key>". The proxy answers
                401 without one (measured 2026-11-07). Blank or whitespace-only
                is treated as absent, matching cortex_utils' LLMClient.
        """
        self.litellm_url = litellm_url.rstrip("/")
        self.model = model
        key = (api_key or "").strip()
        self.http_client = httpx.Client(
            timeout=30.0,
            headers={"Authorization": f"Bearer {key}"} if key else {},
        )

    def parse(self, command: str) -> Optional[ParsedCommand]:
        """Parse natural language command into structured form.

        Args:
            command: Natural language command string

        Returns:
            ParsedCommand or None if parsing fails
        """
        try:
            # Call LLM to parse command
            response = self.http_client.post(
                f"{self.litellm_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": self.PARSE_PROMPT.format(command=command),
                        }
                    ],
                    "temperature": 0.0,  # Deterministic parsing
                },
            )
            response.raise_for_status()

            # Extract JSON from response
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Parse JSON response
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code block
                try:
                    if "```json" in content:
                        json_str = content.split("```json")[1].split("```")[0].strip()
                        parsed = json.loads(json_str)
                    elif "```" in content:
                        json_str = content.split("```")[1].split("```")[0].strip()
                        parsed = json.loads(json_str)
                    else:
                        logger.error(f"Failed to parse LLM response as JSON: {content}")
                        return None
                except (json.JSONDecodeError, IndexError):
                    logger.error(
                        f"Failed to extract or parse JSON from markdown block: {content}",
                        exc_info=True,
                    )
                    return None

            # Validate required fields
            if "action" not in parsed or "target" not in parsed:
                logger.error(f"Missing required fields in parsed command: {parsed}")
                return None

            # Build ParsedCommand
            return ParsedCommand(
                action=parsed["action"],
                target=parsed["target"],
                target_keywords=parsed.get("target_keywords", []),
                parameters=parsed.get("parameters", {}),
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"LLM API error during command parsing: {e}")
            return None
        except Exception as e:
            logger.error(f"Command parsing failed: {e}", exc_info=True)
            return None

    def close(self) -> None:
        """Close HTTP client.

        Should be called during graceful shutdown to ensure clean resource cleanup.
        """
        self.http_client.close()
        logger.info("Closed CommandParser HTTP client")
