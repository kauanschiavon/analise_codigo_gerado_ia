#!/usr/bin/env python3
"""
Reconstrói os arquivos sample_*.py a partir dos prompts e respostas brutas,
preservando a indentação inicial da resposta do modelo.

Não consulta o Ollama e não altera a pasta original.
Executar a partir da raiz do projeto:
    python3 reconstruir_amostras.py
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any


def is_fence(line: str, opening: bool = False) -> bool:
    if opening:
        return bool(re.fullmatch(r"[ \t]*```(?:python|py)?[ \t]*", line, re.I))
    return bool(re.fullmatch(r"[ \t]*```[ \t]*", line))


def extract_completion_preserving_indent(raw_text: str) -> str:
    """
    Remove cercas Markdown, mas NÃO usa strip() na resposta inteira,
    pois isso apagaria os espaços iniciais da primeira linha de código.
    """
    lines = raw_text.splitlines()

    # Remover apenas linhas vazias externas. Não mexer no recuo das linhas de código.
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    opening = next(
        (i for i, line in enumerate(lines) if is_fence(line, opening=True)),
        None,
    )

    if opening is not None:
        closing = next(
            (i for i in range(opening + 1, len(lines)) if is_fence(lines[i])),
            None,
        )
        if closing is not None:
            code_lines = lines[opening + 1 : closing]
        else:
            code_lines = lines[opening + 1 :]
    else:
        code_lines = lines

    # Novamente, retirar somente linhas vazias nas extremidades.
    code_lines = list(code_lines)
    while code_lines and not code_lines[0].strip():
        code_lines.pop(0)
    while code_lines and not code_lines[-1].strip():
        code_lines.pop()

    return "\n".join(code_lines)


def assemble_source(prompt: str, completion: str, entry_point: str) -> str:
    """Junta prompt e continuação; preserva o prompt original."""
    function_pattern = re.compile(
        rf"(?m)^[ \t]*def[ \t]+{re.escape(entry_point)}[ \t]*\("
    )
    match = function_pattern.search(completion)

    if match:
        prompt_function = function_pattern.search(prompt)
        prefix = prompt[: prompt_function.start()] if prompt_function else ""
        pieces = [
            piece.rstrip("\n")
            for piece in (prefix, completion)
            if piece.strip()
        ]
        return "\n\n".join(pieces) + "\n"

    separator = "" if prompt.endswith("\n") else "\n"
    return prompt + separator + completion.rstrip() + "\n"


def syntax_result(source: str) -> tuple[bool, dict[str, Any] | None]:
    try:
        ast.parse(source)
        return True, None
    except SyntaxError as exc:
        return False, {
            "type": type(exc).__name__,
            "message": exc.msg,
            "line": exc.lineno,
            "offset": exc.offset,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="gerados/qwen2.5-coder-7b-automatico-v2",
        help="Pasta original das gerações",
    )
    parser.add_argument(
        "--output",
        default="gerados/qwen2.5-coder-7b-automatico-v2-reconstruido",
        help="Nova pasta para os códigos reconstruídos",
    )
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not source_dir.is_dir():
        print(f"Erro: pasta original não encontrada: {source_dir}")
        return 1
    if output_dir.exists():
        print(f"Erro: a pasta de saída já existe: {output_dir}")
        print("Escolha outro caminho com --output para não sobrescrever resultados.")
        return 1

    manifest_path = source_dir / "manifest.jsonl"
    manifest_by_file: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                raw_path = record.get("completion_file")
                if raw_path:
                    manifest_by_file[Path(raw_path).name] = record
            except json.JSONDecodeError:
                continue

    raw_files = sorted(source_dir.glob("HumanEval_*/sample_*_completion.txt"))
    if not raw_files:
        print(f"Nenhuma resposta bruta *_completion.txt encontrada em {source_dir}")
        return 1

    output_dir.mkdir(parents=True)
    config_path = source_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = {}
    config["reconstruction"] = {
        "method": "preserve_leading_indentation_v1",
        "source_directory": str(source_dir),
        "note": (
            "Arquivos .py reconstruídos das respostas brutas. "
            "Os arquivos originais permanecem intactos."
        ),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    new_manifest: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []

    for raw_path in raw_files:
        problem_dir = raw_path.parent
        prompt_path = problem_dir / "prompt.txt"
        if not prompt_path.exists():
            print(f"Aviso: prompt ausente, pulando {raw_path}")
            continue

        sample_match = re.fullmatch(r"sample_(\d+)_completion\.txt", raw_path.name)
        folder_match = re.fullmatch(r"HumanEval_(\d+)", problem_dir.name)
        if not sample_match or not folder_match:
            continue

        sample_number = int(sample_match.group(1))
        problem_number = int(folder_match.group(1))
        task_id = f"HumanEval/{problem_number}"
        source_name = f"sample_{sample_number}.py"
        new_problem_dir = output_dir / problem_dir.name
        new_problem_dir.mkdir(exist_ok=True)

        prompt = prompt_path.read_text(encoding="utf-8")
        raw_text = raw_path.read_text(encoding="utf-8")
        old_record = manifest_by_file.get(raw_path.name, {})

        # HumanEval tem cinco amostras por problema; copia prompt e raw sem edição.
        target_prompt = new_problem_dir / "prompt.txt"
        if not target_prompt.exists():
            shutil.copy2(prompt_path, target_prompt)
        target_raw = new_problem_dir / raw_path.name
        shutil.copy2(raw_path, target_raw)

        completion = extract_completion_preserving_indent(raw_text)
        entry_point = old_record.get("entry_point", "")
        # entry_point normalmente não consta no manifest; a função do prompt é obtida.
        if not entry_point:
            match = re.search(r"(?m)^[ \t]*def[ \t]+([A-Za-z_]\w*)[ \t]*\(", prompt)
            entry_point = match.group(1) if match else ""

        assembled = assemble_source(prompt, completion, entry_point) if entry_point else (
            prompt + ("" if prompt.endswith("\n") else "\n") + completion.rstrip() + "\n"
        )
        target_source = new_problem_dir / source_name
        target_source.write_text(assembled, encoding="utf-8")

        valid, error = syntax_result(assembled)
        new_record = dict(old_record)
        new_record.update(
            {
                "task_id": task_id,
                "sample": sample_number,
                "completion_file": str(target_raw),
                "assembled_source_file": str(target_source),
                "syntax_valid": valid,
                "syntax_error": error,
                "reconstructed": True,
                "reconstruction_method": "preserve_leading_indentation_v1",
                "previous_syntax_valid": old_record.get("syntax_valid"),
            }
        )
        new_manifest.append(new_record)

        report_rows.append(
            {
                "task_id": task_id,
                "sample": sample_number,
                "syntax_valid_after_reconstruction": valid,
                "previous_syntax_valid": old_record.get("syntax_valid", ""),
                "error_type": error["type"] if error else "",
                "error_message": error["message"] if error else "",
                "error_line": error["line"] if error else "",
                "raw_file": str(target_raw),
                "source_file": str(target_source),
            }
        )

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as file:
        for record in new_manifest:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    valid_count = sum(row["syntax_valid_after_reconstruction"] for row in report_rows)
    total = len(report_rows)
    invalid_count = total - valid_count

    csv_path = output_dir / "relatorio_reconstrucao.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    print("\n=== RECONSTRUÇÃO CONCLUÍDA ===")
    print(f"Respostas brutas processadas: {total}")
    print(f"Arquivos sintaticamente válidos: {valid_count}")
    print(f"Arquivos ainda inválidos: {invalid_count}")
    print(f"Nova pasta: {output_dir}")
    print(f"Relatório: {csv_path}")
    print("\nA pasta original não foi modificada.")
    print("Se ainda houver inválidos, confira os arquivos raw correspondentes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
