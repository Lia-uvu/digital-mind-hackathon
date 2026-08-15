"""Small local chat-model wrapper with repeatable branch sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence


ChatMessage = Mapping[str, str]


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 48
    temperature: float = 0.5
    top_p: float = 0.9

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0


@dataclass(frozen=True)
class GeneratedText:
    text: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]


class LocalChatModel:
    """Own one Hugging Face causal LM and expose deterministic seeded turns."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str,
        device: Any,
        snapshot_checksum: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device
        self.snapshot_checksum = snapshot_checksum

    @classmethod
    def load(
        cls,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
    ) -> "LocalChatModel":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - exercised in real smoke test
            raise RuntimeError(
                "Model dependencies are missing; install requirements.txt first"
            ) from error

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        target = torch.device(device)
        dtype = torch.float16 if target.type in {"mps", "cuda"} else torch.float32
        local_path = Path(model_name)
        snapshot_checksum = (
            local_snapshot_checksum(local_path) if local_path.is_dir() else None
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model.to(target)
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            device=target,
            snapshot_checksum=snapshot_checksum,
        )

    def render_messages(
        self, messages: Sequence[ChatMessage], *, add_generation_prompt: bool = True
    ) -> str:
        return self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        seed: int,
        sampling: SamplingConfig,
    ) -> str:
        return self.generate_detailed(
            messages, seed=seed, sampling=sampling
        ).text

    def generate_detailed(
        self,
        messages: Sequence[ChatMessage],
        *,
        seed: int,
        sampling: SamplingConfig,
    ) -> GeneratedText:
        """Generate once while retaining the exact prompt/completion token boundary."""
        import torch

        torch.manual_seed(seed)
        if self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.manual_seed(seed)

        rendered = self.render_messages(messages)
        encoded = self.tokenizer(rendered, return_tensors="pt").to(self.device)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": sampling.max_new_tokens,
            "do_sample": sampling.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if sampling.do_sample:
            generate_kwargs.update(
                temperature=sampling.temperature,
                top_p=sampling.top_p,
            )
        else:
            # Qwen's bundled generation config contains sampling-only defaults;
            # explicitly clear them so Transformers does not emit a false warning.
            generate_kwargs.update(temperature=None, top_p=None, top_k=None)
        with torch.inference_mode():
            output = self.model.generate(**encoded, **generate_kwargs)
        prompt_length = encoded["input_ids"].shape[1]
        new_tokens = output[0, prompt_length:]
        return GeneratedText(
            text=self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
            prompt_token_ids=tuple(int(token) for token in encoded["input_ids"][0]),
            generated_token_ids=tuple(int(token) for token in new_tokens),
        )

    def metadata(self) -> dict[str, Any]:
        config = self.model.config
        return {
            "name": self.model_name,
            "model_type": getattr(config, "model_type", None),
            "revision": getattr(config, "_commit_hash", None),
            "snapshot_sha256": self.snapshot_checksum,
            "dtype": str(getattr(self.model, "dtype", "unknown")),
            "device": str(self.device),
        }


def sampling_metadata(config: SamplingConfig) -> dict[str, Any]:
    return asdict(config)


def local_snapshot_checksum(path: str | Path) -> str:
    """Hash model weights and the tokenizer/config files that define inference."""
    source = Path(path)
    patterns = (
        "*.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    )
    files = sorted({file for pattern in patterns for file in source.glob(pattern)})
    if not files:
        raise ValueError(f"no model snapshot files found in {source}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()
