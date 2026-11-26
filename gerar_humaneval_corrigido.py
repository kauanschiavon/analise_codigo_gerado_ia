#!/usr/bin/env python3
"""Gera amostras do HumanEval usando chamadas independentes ao Ollama.

Uso esperado na raiz do projeto:
    python gerar_humaneval.py

O script preserva a resposta bruta e também monta um arquivo .py com
(prompt original do HumanEval + continuação gerada pelo modelo).
"""

from __future__ import annotations

import ast
import datetime as dt
import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

# ----------------------- Configuração do experimento -----------------------
OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5-coder:7b"
DATASET = Path("dados/HumanEval.jsonl.gz")
OUTPUT_DIR = Path("gerados/qwen2.5-coder-7b-automatico-v2")

N_PROBLEMS = 20  # Primeiro teste; depois altere para 20.
N_SAMPLES = 5  # Primeiro teste; depois altere para 5.
TEMPERATURE = 0.2
NUM_PREDICT = 512  # Limite fixo de tokens gerados por amostra.
TIMEOUT_SECONDS = 600

SYSTEM_PROMPT = (
    "You are completing a Python function from the HumanEval benchmark. "
    "Continue the code exactly from where the provided prompt ends. "
    "Output only the code continuation required to complete the function. "
    "Do not repeat the existing prompt, function signature, or docstring. "
    "Do not include Markdown fences, explanations, or tests."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_problems(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Não encontrei {path}. Baixe HumanEval.jsonl.gz e coloque-o nessa pasta."
        )

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        problems = [json.loads(line) for line in file if line.strip()]

    if len(problems) < N_PROBLEMS:
        raise ValueError(
            f"O arquivo contém apenas {len(problems)} problemas; "
            f"são necessários pelo menos {N_PROBLEMS}."
        )
    return problems[:N_PROBLEMS]


def get_ollama_info(session: requests.Session) -> dict[str, Any]:
    """Busca versão e modelos locais para registrar o ambiente."""
    info: dict[str, Any] = {}
    try:
        response = session.get(f"{OLLAMA_URL}/api/version", timeout=10)
        response.raise_for_status()
        info["ollama_version"] = response.json().get("version")
    except requests.RequestException:
        info["ollama_version"] = "não foi possível consultar"

    try:
        response = session.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])
        info["local_models"] = [
            {"name": item.get("name"), "digest": item.get("digest")}
            for item in models
        ]
        available_names = {item.get("name") for item in models}
        if MODEL not in available_names:
            raise RuntimeError(
                f"O modelo {MODEL!r} não apareceu em 'ollama list'. "
                f"Modelos encontrados: {sorted(x for x in available_names if x)}"
            )
    except requests.RequestException as exc:
        raise RuntimeError(
            "Não consegui acessar o Ollama em http://localhost:11434. "
            "Abra o aplicativo Ollama e tente novamente."
        ) from exc
    return info


def extract_code_completion(raw_text: str) -> str:
    """Remove cercas Markdown, preservando a resposta bruta em outro arquivo."""
    lines = raw_text.strip().splitlines()
    opening = next(
        (i for i, line in enumerate(lines)
         if re.fullmatch(r"\s*```(?:python|py)?\s*", line, flags=re.IGNORECASE)),
        None,
    )

    if opening is not None:
        closing = next(
            (i for i in range(opening + 1, len(lines))
             if re.fullmatch(r"\s*```\s*", lines[i])),
            None,
        )
        if closing is not None:
            code_lines = lines[opening + 1:closing]
        else:
            # Se o modelo esqueceu a cerca final, ainda aproveita o bloco após a abertura.
            code_lines = lines[opening + 1:]
    else:
        code_lines = lines

    return "\n".join(code_lines).strip("\n")


def assemble_source(prompt: str, completion: str, entry_point: str) -> str:
    """Monta o arquivo final, evitando duplicar a função se o modelo a repetir."""
    function_pattern = re.compile(
        rf"(?m)^\s*def\s+{re.escape(entry_point)}\s*\("
    )
    match = function_pattern.search(completion)

    if match:
        # Se a resposta trouxer a função inteira, mantém os imports do prompt
        # e usa a função gerada, em vez de concatenar duas assinaturas.
        prompt_function = function_pattern.search(prompt)
        prefix = prompt[:prompt_function.start()] if prompt_function else ""
        pieces = [piece.strip("\n") for piece in (prefix, completion) if piece.strip()]
        return "\n\n".join(pieces) + "\n"

    # Formato esperado pelo HumanEval: o modelo completa o prompt recebido.
    separator = "" if prompt.endswith("\n") else "\n"
    return prompt + separator + completion.rstrip() + "\n"


def generate_one(
    session: requests.Session, prompt: str
) -> tuple[str, dict[str, Any]]:
    """Uma chamada sem histórico: não envia mensagens de gerações anteriores."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
        },
    }

    started = time.perf_counter()
    response = session.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    result = response.json()
    content = result.get("message", {}).get("content", "")
    elapsed = time.perf_counter() - started

    if not content.strip():
        raise RuntimeError("O Ollama retornou uma resposta vazia.")

    metadata = {
        "model_returned": result.get("model"),
        "created_at": result.get("created_at"),
        "elapsed_seconds": round(elapsed, 3),
        "total_duration_ns": result.get("total_duration"),
        "prompt_eval_count": result.get("prompt_eval_count"),
        "eval_count": result.get("eval_count"),
        "done_reason": result.get("done_reason"),
    }
    return content, metadata


def main() -> int:
    print("=== Geração HumanEval com Ollama ===")
    print(f"Modelo: {MODEL}")
    print(f"Problemas: {N_PROBLEMS} | Amostras por problema: {N_SAMPLES}")
    print(f"Temperatura: {TEMPERATURE} | num_predict: {NUM_PREDICT}")
    print("Cada chamada será independente, sem histórico de conversa.\n")

    try:
        problems = load_problems(DATASET)
        with requests.Session() as session:
            ollama_info = get_ollama_info(session)

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            config = {
                "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "dataset": str(DATASET),
                "dataset_sha256": sha256_file(DATASET),
                "model": MODEL,
                "ollama_url": OLLAMA_URL,
                "ollama": ollama_info,
                "n_problems": N_PROBLEMS,
                "n_samples_per_problem": N_SAMPLES,
                "temperature": TEMPERATURE,
                "num_predict": NUM_PREDICT,
                "system_prompt": SYSTEM_PROMPT,
                "independent_requests": True,
                "completion_preprocessing": "remove_markdown_fences_v1",
            }
            (OUTPUT_DIR / "config.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            manifest_path = OUTPUT_DIR / "manifest.jsonl"
            errors_path = OUTPUT_DIR / "errors.jsonl"
            total = N_PROBLEMS * N_SAMPLES
            completed = 0

            with manifest_path.open("a", encoding="utf-8") as manifest:
                with errors_path.open("a", encoding="utf-8") as errors:
                    for problem_index, problem in enumerate(problems, start=1):
                        task_id = problem["task_id"]
                        prompt = problem["prompt"]
                        problem_number = int(task_id.rsplit("/", 1)[-1])
                        problem_dir = OUTPUT_DIR / f"HumanEval_{problem_number:03d}"
                        problem_dir.mkdir(parents=True, exist_ok=True)
                        (problem_dir / "prompt.txt").write_text(
                            prompt, encoding="utf-8"
                        )

                        for sample_number in range(1, N_SAMPLES + 1):
                            completed += 1
                            stem = f"sample_{sample_number}"
                            raw_path = problem_dir / f"{stem}_completion.txt"
                            source_path = problem_dir / f"{stem}.py"

                            # Não sobrescreve amostras já existentes ao retomar a execução.
                            if raw_path.exists() or source_path.exists():
                                print(
                                    f"[{completed}/{total}] {task_id} amostra "
                                    f"{sample_number}: já existe; pulando."
                                )
                                continue

                            try:
                                completion, generation_info = generate_one(
                                    session, prompt
                                )
                                # Preserva a resposta exata do modelo para auditoria.
                                raw_path.write_text(completion, encoding="utf-8")

                                # Remove cercas Markdown e monta a fonte para avaliação.
                                cleaned_completion = extract_code_completion(completion)
                                assembled_source = assemble_source(
                                    prompt, cleaned_completion, problem["entry_point"]
                                )
                                source_path.write_text(assembled_source, encoding="utf-8")

                                syntax_valid = True
                                syntax_error = None
                                try:
                                    ast.parse(assembled_source)
                                except SyntaxError as exc:
                                    syntax_valid = False
                                    syntax_error = {
                                        "message": exc.msg,
                                        "line": exc.lineno,
                                        "offset": exc.offset,
                                    }

                                record = {
                                    "task_id": task_id,
                                    "sample": sample_number,
                                    "completion_file": str(raw_path),
                                    "assembled_source_file": str(source_path),
                                    "syntax_valid": syntax_valid,
                                    "syntax_error": syntax_error,
                                    "markdown_fence_present_in_raw": bool(re.search(r"(?m)^\s*```", completion)),
                                    **generation_info,
                                    "status": "ok",
                                }
                                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                                manifest.flush()
                                print(
                                    f"[{completed}/{total}] {task_id} amostra "
                                    f"{sample_number}: salva; sintaxe={'válida' if syntax_valid else 'inválida'} ({generation_info['elapsed_seconds']} s)"
                                )
                            except (requests.RequestException, ValueError, RuntimeError) as exc:
                                record = {
                                    "task_id": task_id,
                                    "sample": sample_number,
                                    "error": str(exc),
                                    "status": "error",
                                }
                                errors.write(json.dumps(record, ensure_ascii=False) + "\n")
                                errors.flush()
                                print(
                                    f"[{completed}/{total}] ERRO em {task_id}, "
                                    f"amostra {sample_number}: {exc}",
                                    file=sys.stderr,
                                )

    except (FileNotFoundError, ValueError, RuntimeError, requests.RequestException) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    print("\nGeração finalizada. Confira os arquivos em:")
    print(OUTPUT_DIR)
    print("Respostas brutas: *_completion.txt | Código limpo/montado: sample_*.py")
    print("Consulte manifest.jsonl e errors.jsonl para o registro das execuções.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
