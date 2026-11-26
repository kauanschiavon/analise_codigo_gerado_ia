#!/usr/bin/env python3
"""
Analisa estaticamente as amostras geradas pelo Qwen e as soluções humanas
do HumanEval usando Pylint e Bandit.

Uso a partir da raiz do projeto:
    python3 analisar.py

Teste reduzido:
    python3 analisar.py --limit 1

Opções:
    python3 analisar.py --root gerados/qwen2.5-coder-7b-automatico
    python3 analisar.py --data dados/HumanEval.jsonl.gz

Critérios:
- Códigos gerados que não compilam são registrados como excluídos.
- Soluções humanas são montadas como prompt + canonical_solution.
- Apenas alertas em linhas do corpo da função são contabilizados.
- Assinatura e docstring são excluídas da contagem.
- Pylint: error/fatal/warning = erro_provavel; convention/refactor = estilo.
- Bandit = seguranca.
- O script não executa o código analisado.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = "gerados/qwen2.5-coder-7b-automatico"
DEFAULT_DATA = "dados/HumanEval.jsonl.gz"
TOOL_TIMEOUT_SECONDS = 60


def read_dataset(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado: {path}\n"
            "Verifique se dados/HumanEval.jsonl.gz existe."
        )

    opener = gzip.open if path.suffix == ".gz" else open
    problems: dict[str, dict[str, Any]] = {}
    with opener(path, "rt", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            task_id = item.get("task_id")
            if isinstance(task_id, str):
                problems[task_id] = item
    return problems


def find_function_node(source: str, entry_point: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == entry_point:
                return node
    raise ValueError(f"Função {entry_point!r} não encontrada no código.")


def function_body_info(
    source: str, entry_point: str
) -> tuple[set[int], int]:
    """
    Retorna as linhas pertencentes ao corpo da função (sem docstring)
    e a quantidade de linhas físicas não vazias/não comentadas nesse corpo.
    """
    node = find_function_node(source, entry_point)
    body = list(node.body)

    # Exclui docstring da função, se presente.
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    body_lines: set[int] = set()
    for statement in body:
        start = getattr(statement, "lineno", None)
        end = getattr(statement, "end_lineno", start)
        if start is not None:
            body_lines.update(range(start, (end or start) + 1))

    lines = source.splitlines()
    loc = 0
    for line_number in body_lines:
        if 1 <= line_number <= len(lines):
            stripped = lines[line_number - 1].strip()
            if stripped and not stripped.startswith("#"):
                loc += 1

    return body_lines, loc


def run_tool(command: list[str]) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUT_SECONDS,
            check=False,
        )
        return result.returncode, result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", f"Timeout após {TOOL_TIMEOUT_SECONDS}s"
    except OSError as exc:
        return 127, "", str(exc)


def parse_pylint(path: Path, body_lines: set[int]) -> list[dict[str, Any]]:
    command = [
        "pylint",
        "--score=n",
        "--reports=n",
        "--output-format=json",
        str(path),
    ]
    code, stdout, stderr = run_tool(command)

    try:
        messages = json.loads(stdout) if stdout.strip() else []
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Não foi possível interpretar JSON do Pylint para {path}.\n"
            f"Saída: {stdout[-1000:]}\nErro: {stderr[-1000:]}"
        )

    findings: list[dict[str, Any]] = []
    for message in messages:
        line_number = message.get("line")
        if not isinstance(line_number, int) or line_number not in body_lines:
            continue

        message_type = str(message.get("type", "")).lower()
        if message_type in {"fatal", "error", "warning"}:
            category = "erro_provavel"
        elif message_type in {"convention", "refactor"}:
            category = "estilo"
        else:
            category = "outro"

        findings.append(
            {
                "tool": "pylint",
                "category": category,
                "severity": message_type,
                "code": message.get("message-id", ""),
                "symbol": message.get("symbol", ""),
                "message": message.get("message", ""),
                "line": line_number,
                "confidence": "",
            }
        )
    return findings


def parse_bandit(path: Path, body_lines: set[int]) -> list[dict[str, Any]]:
    command = ["bandit", "-f", "json", "-q", str(path)]
    code, stdout, stderr = run_tool(command)

    try:
        payload = json.loads(stdout) if stdout.strip() else {"results": []}
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Não foi possível interpretar JSON do Bandit para {path}.\n"
            f"Saída: {stdout[-1000:]}\nErro: {stderr[-1000:]}"
        )

    findings: list[dict[str, Any]] = []
    for issue in payload.get("results", []):
        line_number = issue.get("line_number")
        if not isinstance(line_number, int) or line_number not in body_lines:
            continue

        findings.append(
            {
                "tool": "bandit",
                "category": "seguranca",
                "severity": str(issue.get("issue_severity", "")).lower(),
                "code": issue.get("test_id", ""),
                "symbol": issue.get("test_name", ""),
                "message": issue.get("issue_text", ""),
                "line": line_number,
                "confidence": str(issue.get("issue_confidence", "")).lower(),
            }
        )
    return findings


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compara alertas Pylint/Bandit entre Qwen e soluções humanas."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limita o número de problemas para um teste rápido (0 = todos).",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    data_path = Path(args.data).expanduser()

    if not root.is_dir():
        print(f"Erro: pasta de gerações não encontrada: {root}")
        return 1
    if args.limit < 0:
        print("Erro: --limit não pode ser negativo.")
        return 1

    for tool in ("pylint", "bandit"):
        if shutil.which(tool) is None:
            print(f"Erro: {tool} não encontrado.")
            print("Instale com: pip install pylint bandit")
            return 1

    try:
        problems = read_dataset(data_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Erro ao ler HumanEval: {exc}")
        return 1

    generated_files = sorted(root.glob("HumanEval_*/sample_*.py"))
    if not generated_files:
        print(f"Nenhum sample_*.py encontrado em {root}")
        return 1

    output_dir = root / "resultados_analise_estatica"
    human_source_dir = output_dir / "solucoes_humanas"
    human_source_dir.mkdir(parents=True, exist_ok=True)

    # Seleciona apenas problemas efetivamente presentes nos arquivos gerados.
    problem_numbers = sorted(
        {
            int(path.parent.name.replace("HumanEval_", ""))
            for path in generated_files
            if path.parent.name.startswith("HumanEval_")
        }
    )
    if args.limit:
        problem_numbers = problem_numbers[: args.limit]

    selected_ids = {f"HumanEval/{number}" for number in problem_numbers}
    generated_files = [
        path for path in generated_files
        if f"HumanEval/{int(path.parent.name.replace('HumanEval_', ''))}" in selected_ids
    ]

    candidates: list[dict[str, Any]] = []

    # Grupo gerado pelo modelo.
    for source_path in generated_files:
        problem_number = int(source_path.parent.name.replace("HumanEval_", ""))
        task_id = f"HumanEval/{problem_number}"
        problem = problems.get(task_id)
        if problem is None:
            continue

        entry_point = problem["entry_point"]
        source = source_path.read_text(encoding="utf-8")

        try:
            compile(source, str(source_path), "exec")
            body_lines, loc = function_body_info(source, entry_point)
            candidates.append(
                {
                    "group": "qwen",
                    "task_id": task_id,
                    "sample": int(source_path.stem.replace("sample_", "")),
                    "source_path": source_path,
                    "entry_point": entry_point,
                    "body_lines": body_lines,
                    "loc_body": loc,
                    "compile_valid": True,
                }
            )
        except (SyntaxError, ValueError) as exc:
            candidates.append(
                {
                    "group": "qwen",
                    "task_id": task_id,
                    "sample": int(source_path.stem.replace("sample_", "")),
                    "source_path": source_path,
                    "entry_point": entry_point,
                    "body_lines": set(),
                    "loc_body": 0,
                    "compile_valid": False,
                    "compile_error": str(exc),
                }
            )

    # Grupo humano: prompt + canonical_solution oficial, sem executar nada.
    for problem_number in problem_numbers:
        task_id = f"HumanEval/{problem_number}"
        problem = problems.get(task_id)
        if problem is None:
            print(f"Aviso: {task_id} não foi encontrado no dataset.")
            continue

        human_source = problem["prompt"] + problem["canonical_solution"]
        if not human_source.endswith("\n"):
            human_source += "\n"

        human_path = human_source_dir / f"HumanEval_{problem_number:03d}.py"
        human_path.write_text(human_source, encoding="utf-8")

        entry_point = problem["entry_point"]
        try:
            compile(human_source, str(human_path), "exec")
            body_lines, loc = function_body_info(human_source, entry_point)
            candidates.append(
                {
                    "group": "humano",
                    "task_id": task_id,
                    "sample": "canonical",
                    "source_path": human_path,
                    "entry_point": entry_point,
                    "body_lines": body_lines,
                    "loc_body": loc,
                    "compile_valid": True,
                }
            )
        except (SyntaxError, ValueError) as exc:
            candidates.append(
                {
                    "group": "humano",
                    "task_id": task_id,
                    "sample": "canonical",
                    "source_path": human_path,
                    "entry_point": entry_point,
                    "body_lines": set(),
                    "loc_body": 0,
                    "compile_valid": False,
                    "compile_error": str(exc),
                }
            )

    detail_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    tool_errors: list[dict[str, Any]] = []

    total_candidates = len(candidates)
    print("=== Análise estática HumanEval ===")
    print(f"Problemas selecionados: {len(problem_numbers)}")
    print(f"Arquivos candidatos: {total_candidates}")
    print("Executando Pylint e Bandit (análise estática, sem executar os programas).\n")

    for index, candidate in enumerate(candidates, start=1):
        group = candidate["group"]
        task_id = candidate["task_id"]
        sample = candidate["sample"]
        source_path = candidate["source_path"]
        row_findings: list[dict[str, Any]] = []

        if not candidate["compile_valid"]:
            sample_rows.append(
                {
                    "group": group,
                    "task_id": task_id,
                    "sample": sample,
                    "compile_valid": False,
                    "compile_error": candidate.get("compile_error", ""),
                    "loc_body": 0,
                    "pylint_count": 0,
                    "bandit_count": 0,
                    "error_provavel_count": 0,
                    "seguranca_count": 0,
                    "estilo_count": 0,
                    "total_alerts": 0,
                    "alerts_per_100_loc": "",
                    "analysis_status": "skipped_compile_error",
                }
            )
            print(f"[{index}/{total_candidates}] {group} {task_id} {sample}: ignorado (não compila)")
            continue

        try:
            row_findings.extend(
                parse_pylint(source_path, candidate["body_lines"])
            )
        except Exception as exc:
            tool_errors.append(
                {
                    "group": group,
                    "task_id": task_id,
                    "sample": sample,
                    "tool": "pylint",
                    "error": str(exc),
                }
            )

        try:
            row_findings.extend(
                parse_bandit(source_path, candidate["body_lines"])
            )
        except Exception as exc:
            tool_errors.append(
                {
                    "group": group,
                    "task_id": task_id,
                    "sample": sample,
                    "tool": "bandit",
                    "error": str(exc),
                }
            )

        for finding in row_findings:
            detail_rows.append(
                {
                    "group": group,
                    "task_id": task_id,
                    "sample": sample,
                    "tool": finding["tool"],
                    "category": finding["category"],
                    "severity": finding["severity"],
                    "code": finding["code"],
                    "symbol": finding["symbol"],
                    "message": finding["message"],
                    "line": finding["line"],
                    "confidence": finding["confidence"],
                    "source_file": str(source_path),
                }
            )

        counts = Counter(item["category"] for item in row_findings)
        pylint_count = sum(item["tool"] == "pylint" for item in row_findings)
        bandit_count = sum(item["tool"] == "bandit" for item in row_findings)
        loc = candidate["loc_body"]
        total_alerts = len(row_findings)

        sample_rows.append(
            {
                "group": group,
                "task_id": task_id,
                "sample": sample,
                "compile_valid": True,
                "compile_error": "",
                "loc_body": loc,
                "pylint_count": pylint_count,
                "bandit_count": bandit_count,
                "error_provavel_count": counts["erro_provavel"],
                "seguranca_count": counts["seguranca"],
                "estilo_count": counts["estilo"],
                "total_alerts": total_alerts,
                "alerts_per_100_loc": round(total_alerts / loc * 100, 4) if loc else 0,
                "analysis_status": "ok",
            }
        )

        print(
            f"[{index}/{total_candidates}] {group} {task_id} {sample}: "
            f"{total_alerts} alertas; {loc} linhas de corpo"
        )

    # Agregação por grupo e categoria.
    aggregate: dict[str, dict[str, Any]] = {}
    for group in ("qwen", "humano"):
        group_rows = [
            row for row in sample_rows
            if row["group"] == group and row["analysis_status"] == "ok"
        ]
        loc_total = sum(int(row["loc_body"]) for row in group_rows)
        alert_total = sum(int(row["total_alerts"]) for row in group_rows)
        aggregate[group] = {
            "samples_analyzed": len(group_rows),
            "samples_skipped": sum(
                row["group"] == group and row["analysis_status"] != "ok"
                for row in sample_rows
            ),
            "loc_body_total": loc_total,
            "pylint_alerts": sum(int(row["pylint_count"]) for row in group_rows),
            "bandit_alerts": sum(int(row["bandit_count"]) for row in group_rows),
            "error_provavel_alerts": sum(
                int(row["error_provavel_count"]) for row in group_rows
            ),
            "security_alerts": sum(
                int(row["seguranca_count"]) for row in group_rows
            ),
            "style_alerts": sum(
                int(row["estilo_count"]) for row in group_rows
            ),
            "total_alerts": alert_total,
            "alerts_per_100_loc": round(alert_total / loc_total * 100, 4)
            if loc_total else 0,
        }

    aggregate_rows: list[dict[str, Any]] = []
    for group, values in aggregate.items():
        aggregate_rows.append({"group": group, **values})

    write_csv(output_dir / "alertas_detalhados.csv", detail_rows)
    write_csv(output_dir / "resumo_por_amostra.csv", sample_rows)
    write_csv(output_dir / "resumo_por_grupo.csv", aggregate_rows)
    write_csv(output_dir / "erros_ferramentas.csv", tool_errors)

    summary = {
        "root": str(root),
        "dataset": str(data_path),
        "problems": [f"HumanEval/{n}" for n in problem_numbers],
        "classification": {
            "pylint_error_fatal_warning": "erro_provavel",
            "pylint_convention_refactor": "estilo",
            "bandit": "seguranca",
            "filter": (
                "Somente alertas cuja linha está dentro do corpo da função; "
                "assinatura e docstring são excluídas."
            ),
        },
        "groups": aggregate,
        "tool_errors_count": len(tool_errors),
        "tool_errors": tool_errors,
        "limitations": [
            "Alertas estáticos são indícios, não prova automática de defeito.",
            "Bandit detecta padrões de segurança, não a ausência de vulnerabilidades.",
            "Alertas devem ser classificados manualmente para estimar precisão.",
            "A densidade usa linhas não vazias e não comentadas no corpo da função.",
            "A comparação exploratória usa 100 gerações e 20 soluções humanas.",
        ],
    }

    summary_path = output_dir / "resumo_geral.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== RESUMO DA ANÁLISE ESTÁTICA ===")
    for group, values in aggregate.items():
        print(f"\nGrupo: {group}")
        print(f"  Amostras analisadas: {values['samples_analyzed']}")
        print(f"  Linhas de corpo: {values['loc_body_total']}")
        print(f"  Pylint: {values['pylint_alerts']}")
        print(f"  Bandit: {values['bandit_alerts']}")
        print(f"  Erro provável: {values['error_provavel_alerts']}")
        print(f"  Segurança: {values['security_alerts']}")
        print(f"  Estilo: {values['style_alerts']}")
        print(f"  Total: {values['total_alerts']}")
        print(f"  Alertas por 100 linhas: {values['alerts_per_100_loc']}")

    print(f"\nErros das ferramentas: {len(tool_errors)}")
    print(f"Relatórios salvos em: {output_dir}")

    if tool_errors:
        print(
            "Atenção: confira erros_ferramentas.csv. "
            "A análise está incompleta para arquivos com falha de ferramenta."
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
