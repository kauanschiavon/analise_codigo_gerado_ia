#!/usr/bin/env python3
"""
Valida a compilação real dos arquivos sample_*.py com compile().

Diferentemente de ast.parse(), compile() também detecta erros de contexto
como "return outside function".

Uso:
    python3 validar.py

Ou:
    python3 validar.py --root gerados/qwen2.5-coder-7b-automatico

O script não modifica os arquivos originais.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ROOT = "gerados/qwen2.5-coder-7b-automatico"


def load_manifest(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                task_id = item.get("task_id")
                sample = item.get("sample")
                if isinstance(task_id, str) and isinstance(sample, int):
                    records[(task_id, sample)] = item
            except json.JSONDecodeError:
                print(f"Aviso: JSON inválido no manifesto, linha {line_number}.")
    return records


def identify_sample(path: Path) -> tuple[str, int] | None:
    folder = re.fullmatch(r"HumanEval_(\d+)", path.parent.name)
    sample = re.fullmatch(r"sample_(\d+)\.py", path.name)
    if not folder or not sample:
        return None
    return f"HumanEval/{int(folder.group(1))}", int(sample.group(1))


def validate_with_compile(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "compile_valid": False,
        "error_type": "",
        "error_message": "",
        "error_line": "",
        "error_offset": "",
    }
    try:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        result["compile_valid"] = True
    except SyntaxError as exc:
        result.update({
            "error_type": type(exc).__name__,
            "error_message": exc.msg or str(exc),
            "error_line": exc.lineno or "",
            "error_offset": exc.offset or "",
        })
    except (OSError, UnicodeDecodeError) as exc:
        result.update({
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"Erro: pasta não encontrada: {root}")
        return 1

    manifest = load_manifest(root / "manifest.jsonl")
    sample_files = sorted(root.glob("HumanEval_*/sample_*.py"))
    if not sample_files:
        print(f"Nenhum sample_*.py encontrado em {root}")
        return 1

    rows: list[dict[str, Any]] = []
    for source_path in sample_files:
        identity = identify_sample(source_path)
        if identity is None:
            continue

        task_id, sample_number = identity
        outcome = validate_with_compile(source_path)
        manifest_record = manifest.get((task_id, sample_number), {})

        rows.append({
            "task_id": task_id,
            "sample": sample_number,
            "source_file": str(source_path),
            "compile_valid": outcome["compile_valid"],
            "error_type": outcome["error_type"],
            "error_message": outcome["error_message"],
            "error_line": outcome["error_line"],
            "error_offset": outcome["error_offset"],
            "previous_ast_valid": manifest_record.get("syntax_valid", ""),
            "completion_file": manifest_record.get("completion_file", ""),
        })

    if not rows:
        print("Nenhuma amostra válida para processamento.")
        return 1

    valid = sum(bool(row["compile_valid"]) for row in rows)
    invalid_rows = [row for row in rows if not row["compile_valid"]]
    total = len(rows)
    invalid_rate = 100 * len(invalid_rows) / total

    report_dir = root / "relatorios_validacao"
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "validacao_compilacao.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    txt_path = report_dir / "erros_compilacao.txt"
    with txt_path.open("w", encoding="utf-8") as file:
        file.write("AMOSTRAS QUE FALHARAM NA COMPILAÇÃO DO PYTHON\n")
        file.write("=" * 52 + "\n\n")
        if not invalid_rows:
            file.write("Nenhuma amostra falhou na compilação.\n")
        for row in invalid_rows:
            file.write(
                f"{row['task_id']} | amostra {row['sample']}\n"
                f"Arquivo: {row['source_file']}\n"
                f"Erro: {row['error_type']}: {row['error_message']}\n"
                f"Linha: {row['error_line']} | coluna: {row['error_offset']}\n"
                f"Resposta bruta: {row['completion_file']}\n\n"
            )

    by_problem = dict(Counter(row["task_id"] for row in invalid_rows))
    summary = {
        "total_files_checked": total,
        "compile_valid": valid,
        "compile_invalid": len(invalid_rows),
        "compile_invalid_rate_percent": round(invalid_rate, 2),
        "invalid_by_problem": by_problem,
        "validation_method": "compile(source, filename, 'exec')",
        "note": (
            "compile() detecta erros de compilação/contexto, incluindo "
            "'return outside function'. Isso não comprova correção funcional."
        ),
    }

    json_path = report_dir / "resumo_compilacao.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== VALIDAÇÃO DE COMPILAÇÃO CONCLUÍDA ===")
    print(f"Arquivos verificados: {total}")
    print(f"Compilam: {valid}")
    print(f"Não compilam: {len(invalid_rows)}")
    print(f"Taxa de não compilação: {invalid_rate:.2f}%")
    print("\nErros por problema:")
    for task_id, count in sorted(
        by_problem.items(), key=lambda item: int(item[0].split("/")[-1])
    ):
        print(f"  {task_id}: {count}")

    print("\nRelatórios:")
    print(f"  {csv_path}")
    print(f"  {txt_path}")
    print(f"  {json_path}")
    print("\nNenhum arquivo .py original foi modificado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
