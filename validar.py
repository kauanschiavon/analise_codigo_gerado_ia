#!/usr/bin/env python3
"""
Valida a sintaxe dos arquivos Python gerados no piloto HumanEval.

Uso, a partir da raiz do projeto:
    python3 validar.py

Ou informando outra pasta:
    python3 validar.py --root gerados/qwen2.5-coder-7b-automatico-v2

O script NÃO altera os arquivos .py. Ele cria relatórios CSV, TXT e JSON.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load_manifest(manifest_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Carrega os registros do manifest.jsonl, se existir."""
    records: dict[tuple[str, int], dict[str, Any]] = {}

    if not manifest_path.exists():
        print(f"Aviso: manifesto não encontrado: {manifest_path}")
        return records

    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                task_id = item.get("task_id")
                sample = item.get("sample")
                if isinstance(task_id, str) and isinstance(sample, int):
                    records[(task_id, sample)] = item
            except json.JSONDecodeError as exc:
                print(
                    f"Aviso: linha {line_number} inválida no manifesto: {exc}"
                )

    return records


def identify_file(path: Path) -> tuple[str, int] | None:
    """Extrai HumanEval/N e número da amostra a partir do caminho."""
    folder_match = re.fullmatch(r"HumanEval_(\d+)", path.parent.name)
    sample_match = re.fullmatch(r"sample_(\d+)\.py", path.name)

    if not folder_match or not sample_match:
        return None

    problem_number = int(folder_match.group(1))
    sample_number = int(sample_match.group(1))
    return f"HumanEval/{problem_number}", sample_number


def validate_file(path: Path) -> dict[str, Any]:
    """Valida o arquivo com ast.parse e devolve detalhes do resultado."""
    result: dict[str, Any] = {
        "syntax_valid": False,
        "error_type": "",
        "error_message": "",
        "error_line": "",
        "error_offset": "",
    }

    try:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        result["syntax_valid"] = True
    except SyntaxError as exc:
        result.update(
            {
                "error_type": type(exc).__name__,
                "error_message": exc.msg or str(exc),
                "error_line": exc.lineno or "",
                "error_offset": exc.offset or "",
            }
        )
    except (OSError, UnicodeDecodeError) as exc:
        result.update(
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida a sintaxe das amostras Python do HumanEval."
    )
    parser.add_argument(
        "--root",
        default="gerados/qwen2.5-coder-7b-automatico-v2",
        help="Pasta raiz das gerações (padrão: %(default)s)",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"Erro: pasta das gerações não encontrada: {root}")
        print("Execute o comando a partir da raiz do projeto ou use --root.")
        return 1

    manifest = load_manifest(root / "manifest.jsonl")
    python_files = sorted(
        path
        for path in root.glob("HumanEval_*/sample_*.py")
        if "__pycache__" not in path.parts
    )

    if not python_files:
        print(f"Nenhum arquivo sample_*.py encontrado em: {root}")
        return 1

    rows: list[dict[str, Any]] = []

    for source_path in python_files:
        identified = identify_file(source_path)
        if identified is None:
            continue

        task_id, sample_number = identified
        manifest_item = manifest.get((task_id, sample_number), {})
        validation = validate_file(source_path)

        completion_file = manifest_item.get("completion_file", "")
        if completion_file:
            completion_path = Path(completion_file)
            if not completion_path.is_absolute():
                # O manifesto guarda caminhos relativos à raiz do projeto.
                project_relative = Path.cwd() / completion_path
                if project_relative.exists():
                    completion_file = str(project_relative)
                else:
                    candidate = root / source_path.parent.name / completion_path.name
                    if candidate.exists():
                        completion_file = str(candidate)

        rows.append(
            {
                "task_id": task_id,
                "sample": sample_number,
                "source_file": str(source_path),
                "completion_file": completion_file,
                "syntax_valid": validation["syntax_valid"],
                "error_type": validation["error_type"],
                "error_message": validation["error_message"],
                "error_line": validation["error_line"],
                "error_offset": validation["error_offset"],
                "manifest_syntax_valid": manifest_item.get("syntax_valid", ""),
                "markdown_fence_present_in_raw": manifest_item.get(
                    "markdown_fence_present_in_raw", ""
                ),
                "model": manifest_item.get("model_returned", ""),
            }
        )

    if not rows:
        print("Nenhum arquivo com padrão HumanEval_NNN/sample_N.py foi encontrado.")
        return 1

    valid_count = sum(bool(row["syntax_valid"]) for row in rows)
    invalid_rows = [row for row in rows if not row["syntax_valid"]]
    total = len(rows)
    invalid_rate = (len(invalid_rows) / total * 100) if total else 0.0

    report_dir = root / "relatorios_validacao"
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "validacao_sintaxe.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    txt_path = report_dir / "amostras_invalidas.txt"
    with txt_path.open("w", encoding="utf-8") as file:
        file.write("RELATÓRIO DE AMOSTRAS COM SINTAXE INVÁLIDA\n")
        file.write("=" * 48 + "\n\n")
        if not invalid_rows:
            file.write("Nenhuma amostra sintaticamente inválida foi encontrada.\n")
        else:
            for row in invalid_rows:
                file.write(
                    f"{row['task_id']} | amostra {row['sample']}\n"
                    f"Arquivo: {row['source_file']}\n"
                    f"Erro: {row['error_type']}: {row['error_message']}\n"
                    f"Linha: {row['error_line']} | coluna: {row['error_offset']}\n"
                    f"Resposta bruta: {row['completion_file']}\n\n"
                )

    error_groups = Counter(
        f"{row['task_id']}: {row['error_message']}" for row in invalid_rows
    )
    summary = {
        "generation_directory": str(root),
        "total_files_checked": total,
        "syntax_valid": valid_count,
        "syntax_invalid": len(invalid_rows),
        "invalid_rate_percent": round(invalid_rate, 2),
        "invalid_by_problem": dict(
            sorted(
                Counter(row["task_id"] for row in invalid_rows).items(),
                key=lambda item: int(item[0].split("/")[-1]),
            )
        ),
        "error_groups": dict(error_groups),
        "note": (
            "A validação sintática não comprova correção funcional. "
            "Investigue a resposta bruta e o prompt dos casos inválidos antes "
            "de atribuir a falha exclusivamente ao modelo."
        ),
    }

    json_path = report_dir / "resumo_validacao.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== VALIDAÇÃO SINTÁTICA CONCLUÍDA ===")
    print(f"Arquivos verificados: {total}")
    print(f"Válidos: {valid_count}")
    print(f"Inválidos: {len(invalid_rows)}")
    print(f"Taxa de inválidos: {invalid_rate:.2f}%")
    print("\nInválidos por problema:")
    if summary["invalid_by_problem"]:
        for task_id, count in summary["invalid_by_problem"].items():
            print(f"  {task_id}: {count}")
    else:
        print("  Nenhum")

    print("\nRelatórios criados em:")
    print(f"  {csv_path}")
    print(f"  {txt_path}")
    print(f"  {json_path}")
    print("\nOs arquivos .py originais não foram modificados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
