"""
LLM Provider abstraction layer for GenEscape.
Supports OpenAI (GPT-4o + DALL-E 3) and Google Gemini (gemini-2.0-flash + Imagen 3).
"""

import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import requests

from genescape.config import (
    LLMProvider,
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    OPENAI_TEXT_MODEL,
    OPENAI_IMAGE_MODEL,
    GEMINI_TEXT_MODEL,
    GEMINI_IMAGE_MODEL,
)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """Text-only chat completion."""
        pass

    @abstractmethod
    def vision_chat(self, messages: list[dict], image_path: Optional[str] = None) -> str:
        """Multimodal chat with optional image."""
        pass

    @abstractmethod
    def generate_image(self, prompt: str, output_path: str) -> str:
        """Generate an image from a prompt, save to output_path, return path."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-backed provider using GPT-4o and DALL-E 3."""

    def __init__(self) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai>=1.0.0")

        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it with: export OPENAI_API_KEY=your_key_here"
            )
        self._client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.text_model = OPENAI_TEXT_MODEL
        self.image_model = OPENAI_IMAGE_MODEL

    def chat(self, messages: list[dict]) -> str:
        """Text-only chat completion using GPT-4o."""
        response = self._client.chat.completions.create(
            model=self.text_model,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    def vision_chat(self, messages: list[dict], image_path: Optional[str] = None) -> str:
        """Multimodal chat; encodes image as base64 if provided."""
        if image_path is not None:
            # Read and encode the image
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            suffix = path.suffix.lower().lstrip(".")
            mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                        "gif": "image/gif", "webp": "image/webp"}
            mime_type = mime_map.get(suffix, "image/png")

            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Append image content block to the last user message
            enriched_messages = []
            for i, msg in enumerate(messages):
                if i == len(messages) - 1 and msg.get("role") == "user":
                    original_content = msg["content"]
                    if isinstance(original_content, str):
                        original_content = [{"type": "text", "text": original_content}]
                    enriched_messages.append({
                        "role": "user",
                        "content": original_content + [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}",
                                    "detail": "high",
                                },
                            }
                        ],
                    })
                else:
                    enriched_messages.append(msg)
        else:
            enriched_messages = messages

        response = self._client.chat.completions.create(
            model=self.text_model,
            messages=enriched_messages,
        )
        return response.choices[0].message.content or ""

    def generate_image(self, prompt: str, output_path: str) -> str:
        """Generate image with DALL-E 3, download and save to output_path."""
        response = self._client.images.generate(
            model=self.image_model,
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url

        # Download the image
        img_response = requests.get(image_url, timeout=60)
        img_response.raise_for_status()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(img_response.content)

        return output_path


class GeminiProvider(BaseLLMProvider):
    """Google Gemini-backed provider using gemini-2.0-flash and Imagen 3.

    Uses the new `google-genai` SDK (pip install google-genai).
    """

    def __init__(self) -> None:
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Run: pip install google-genai>=1.0.0"
            )

        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Export it with: export GOOGLE_API_KEY=your_key_here"
            )

        self._client = genai.Client(api_key=GOOGLE_API_KEY)
        self._types = genai_types
        self.text_model_name = GEMINI_TEXT_MODEL
        self.image_model_name = GEMINI_IMAGE_MODEL

    def chat(self, messages: list[dict]) -> str:
        """Text-only chat using Gemini."""
        contents = self._messages_to_contents(messages)
        response = self._client.models.generate_content(
            model=self.text_model_name,
            contents=contents,
        )
        return response.text or ""

    def vision_chat(self, messages: list[dict], image_path: Optional[str] = None) -> str:
        """Multimodal chat using Gemini; includes image bytes if path provided."""
        contents = self._messages_to_contents(messages)

        if image_path is not None:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            suffix = path.suffix.lower().lstrip(".")
            mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                        "png": "image/png", "webp": "image/webp"}
            mime_type = mime_map.get(suffix, "image/png")

            with open(image_path, "rb") as f:
                image_bytes = f.read()

            # Append image part to last user content block
            image_part = self._types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            if contents and hasattr(contents[-1], "parts"):
                contents[-1].parts.append(image_part)
            else:
                contents.append(
                    self._types.Content(role="user", parts=[image_part])
                )

        response = self._client.models.generate_content(
            model=self.text_model_name,
            contents=contents,
        )
        return response.text or ""

    def generate_image(self, prompt: str, output_path: str) -> str:
        """Generate image using the configured model.

        - Imagen models (name starts with 'imagen-'): use generate_images API.
        - Gemini models: use generate_content with IMAGE response modality.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if self.image_model_name.startswith("imagen-"):
            return self._generate_with_imagen(prompt, output_path)
        return self._generate_with_gemini(prompt, output_path)

    def _generate_with_imagen(self, prompt: str, output_path: str) -> str:
        """Call the Imagen generateImages API."""
        response = self._client.models.generate_images(
            model=self.image_model_name,
            prompt=prompt,
            config=self._types.GenerateImagesConfig(number_of_images=1),
        )
        if not response.generated_images:
            raise RuntimeError("Imagen returned no images.")
        img = response.generated_images[0]
        # Try known attribute names across SDK versions
        image_bytes = (
            getattr(img, "image_bytes", None)
            or getattr(img.image, "image_bytes", None)
            or getattr(img.image, "image_data", None)
        )
        if image_bytes is None:
            raise RuntimeError(f"Cannot find image bytes in Imagen response: {dir(img)}")
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        return output_path

    def _generate_with_gemini(self, prompt: str, output_path: str) -> str:
        """Call generateContent with IMAGE response modality."""
        response = self._client.models.generate_content(
            model=self.image_model_name,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                with open(output_path, "wb") as f:
                    f.write(part.inline_data.data)
                return output_path
        raise RuntimeError(f"{self.image_model_name} returned no image parts.")

    def _messages_to_contents(self, messages: list[dict]) -> list:
        """Convert OpenAI-style messages to google-genai Content objects."""
        contents = []
        system_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )

            if role == "system":
                # Prepend system instructions into the first user message
                system_parts.append(content)
            else:
                gemini_role = "model" if role == "assistant" else "user"
                text = content
                if system_parts and gemini_role == "user":
                    text = "[Instructions]\n" + "\n".join(system_parts) + "\n\n" + text
                    system_parts = []
                contents.append(
                    self._types.Content(
                        role=gemini_role,
                        parts=[self._types.Part.from_text(text=text)],
                    )
                )

        # If only system messages were provided, wrap as a user turn
        if system_parts:
            contents.append(
                self._types.Content(
                    role="user",
                    parts=[self._types.Part.from_text(text="\n".join(system_parts))],
                )
            )

        return contents


def get_provider(provider: LLMProvider) -> BaseLLMProvider:
    """Factory function: return appropriate provider instance."""
    if provider == LLMProvider.OPENAI:
        return OpenAIProvider()
    if provider == LLMProvider.GEMINI:
        return GeminiProvider()
    raise ValueError(f"Unknown provider: {provider}")  # type: ignore[unreachable]
