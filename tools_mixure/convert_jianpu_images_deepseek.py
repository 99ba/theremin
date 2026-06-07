from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".bmp":
        return "image/bmp"
    return "image/jpeg"


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_mime_type(path)};base64,{encoded}"


class ModelJsonError(ValueError):
    def __init__(self, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text


def _extract_json(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ModelJsonError("Model response did not contain a JSON object.", text)
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ModelJsonError(f"Model response contained invalid JSON: {exc}", text) from exc


def _message_content(result: dict) -> str:
    message = result["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text") or ""))
        joined = "\n".join(texts).strip()
        if joined:
            return joined
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    return json.dumps(message, ensure_ascii=False)


def call_qwenvl_image(
    api_key: str,
    prompt: str,
    image_path: Path,
    model: str,
    base_url: str,
    timeout: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                ],
            }
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"QwenVL HTTP {exc.code}: {body}") from exc

    content = _message_content(result)
    return _extract_json(content)


def call_qwen_text(
    api_key: str,
    prompt: str,
    text: str,
    source_name: str,
    model: str,
    base_url: str,
    timeout: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"{prompt}\n\nSource filename: {source_name}\n\nPlain-text jianpu input:\n{text}",
            }
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen HTTP {exc.code}: {body}") from exc

    content = _message_content(result)
    return _extract_json(content)


def convert_directory(args: argparse.Namespace) -> None:
    api_key = (
        args.api_key
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or os.environ.get("QWENVL_API_KEY")
    )
    if not api_key:
        raise SystemExit("Set DASHSCOPE_API_KEY / QWEN_API_KEY or pass --api-key.")
    api_key = api_key.strip().strip('"').strip("'")
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise SystemExit(
            "The QwenVL API key contains non-ASCII characters. "
            "Set it to the real API key only, not the Chinese placeholder text."
        ) from exc

    prompt = (
        "严格要求：最终回答必须只包含一个 JSON 对象。"
        "不要输出思考过程、解释、Markdown、代码块或任何 JSON 之外的文字。\n\n"
        + args.prompt.read_text(encoding="utf-8")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.text_dir is not None:
        inputs = sorted(
            path for path in args.text_dir.iterdir() if path.suffix.lower() in {".txt", ".md"}
        )
        input_mode = "text"
        if not inputs:
            raise SystemExit(f"No .txt or .md files found in {args.text_dir}")
    else:
        inputs = sorted(path for path in args.image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        input_mode = "image"
        if not inputs:
            raise SystemExit(f"No supported images found in {args.image_dir}")

    for input_path in inputs:
        output_path = args.output_dir / f"{input_path.stem}_player_guide.json"
        if output_path.exists() and not args.overwrite:
            print(f"skip existing: {output_path}")
            continue
        print(f"converting: {input_path}")
        try:
            if input_mode == "text":
                data = call_qwen_text(
                    api_key=api_key,
                    prompt=prompt,
                    text=input_path.read_text(encoding="utf-8"),
                    source_name=input_path.name,
                    model=args.model,
                    base_url=args.base_url,
                    timeout=args.timeout,
                )
                data.setdefault("source", f"qwen text conversion: {input_path.name}")
            else:
                data = call_qwenvl_image(
                    api_key=api_key,
                    prompt=f"{prompt}\n\nImage filename: {input_path.name}",
                    image_path=input_path,
                    model=args.model,
                    base_url=args.base_url,
                    timeout=args.timeout,
                )
                data.setdefault("source", f"qwenvl image conversion: {input_path.name}")
        except ModelJsonError as exc:
            raw_path = args.output_dir / f"{input_path.stem}_model_response.raw.txt"
            raw_path.write_text(exc.raw_text, encoding="utf-8")
            raise RuntimeError(f"{exc}. Raw model response saved to {raw_path}") from exc
        output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert jianpu score images to player-guide JSON using QwenVL.")
    parser.add_argument("image_dir", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--text-dir", type=Path, default=None, help="Use plain-text .txt/.md jianpu files instead of images.")
    parser.add_argument("--output-dir", type=Path, default=Path("converted_player_guides"))
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path(__file__).with_name("jianpu_to_player_guide_prompt.md"),
    )
    parser.add_argument("--model", default=os.environ.get("QWEN_VL_MODEL", "qwen-vl-plus"))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        convert_directory(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
