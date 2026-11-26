#!/usr/bin/env python3
"""
Executa os testes oficiais do HumanEval para as amostras geradas.

Uso, a partir da raiz do projeto:
    python3 testar.py

Opções:
    python3 testar.py --root gerados/qwen2.5-coder-7b-automatico
    python3 testar.py --data dados/HumanEval.jsonl.gz --timeout 3

Os códigos são executados em containers Docker efêmeros, sem rede,
com filesystem somente leitura e limites de recursos. Docker reduz o risco,
mas nenhum sandbox deve ser considerado invulnerável.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


DEFAULT_ROOT = "gerados/qwen2.5-coder-7b-automatico"
DEFAULT_DATA = "dados/HumanEval.jsonl.gz"
DEFAULT_IMAGE = "python:3.11-slim"
DEFAULT_TIMEOUT = 3.0
OUTER_TIMEOUT = 12.0


RUNNER_SOURCE = r"""
import contextlib
import json
import os
import signal
from pathlib import Path

def time_limit_handler(signum, frame):
    raise TimeoutError("Tempo limite do teste excedido")

case = json.loads(Path("/work/case.json").read_text(encoding="utf-8"))
candidate_source = Path("/work/candidate.py").read_text(encoding="utf-8")

signal.signal(signal.SIGALRM, time_limit_handler)
signal.setitimer(signal.ITIMER_REAL, float(case["timeout_seconds"]))

namespace = {"__name__": "__main__"}

try:
    # Silencia a saída normal do programa avaliado.
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(compile(candidate_source, "candidate.py", "exec"), namespace)
            exec(compile(case["test"], "humaneval_test.py", "exec"), namespace)

            candidate = namespace.get(case["entry_point"])
            check = namespace.get("check")

            if not callable(candidate):
                raise NameError(
                    "A função esperada não foi encontrada: " + case["entry_point"]
                )
            if not callable(check):
                raise NameError("A função check(candidate) não foi encontrada")

            check(candidate)

    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({"status": "passed"}))
except TimeoutError as exc:
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({
        "status": "timeout",
        "error_type": "TimeoutError",
        "error_message": str(exc)[:500],
    }))
except BaseException as exc:
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:500],
    }))
"""


def load_humaneval(path: Path) -> dict[str, dict[str, Any]]:
    """Carrega o dataset JSONL oficial, compactado ou não."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado: {path}\n"
            "Coloque HumanEval.jsonl.gz em dados/ ou use --data."
        )

    problems: dict[str, dict[str, Any]] = {}
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON inválido no dataset, linha {line_number}: {exc}"
                ) from exc

            task_id = item.get("task_id")
            if isinstance(task_id, str):
                problems[task_id] = item

    return problems


def docker_ready() -> tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "Docker CLI não encontrado. Instale/inicie o Docker Desktop."

    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=15
        )
    except subprocess.TimeoutExpired:
        return False, "O comando docker info excedeu 15 segundos."

    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        return False, (
            "Docker não parece estar em execução. Abra o Docker Desktop. "
            f"Detalhe: {message[-1000:]}"
        )
    return True, ""


def image_exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.returncode == 0


def run_one(
    source_path: Path,
    problem: dict[str, Any],
    image: str,
    timeout_seconds: float,
    project_root: Path,
) -> dict[str, Any]:
    """Executa uma amostra em um container efêmero e restrito."""
    entry_point = problem.get("entry_point")
    test_code = problem.get("test")

    if not isinstance(entry_point, str) or not isinstance(test_code, str):
        return {
            "status": "missing_test_data",
            "error_type": "DatasetError",
            "error_message": "O problema não contém entry_point ou test.",
            "elapsed_seconds": 0.0,
        }

    started = time.monotonic()
    temp_parent = project_root / ".test_tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(
            prefix="humaneval_case_", dir=str(temp_parent)
        ) as temp_name:
            sandbox_dir = Path(temp_name)
            shutil.copy2(source_path, sandbox_dir / "candidate.py")

            case_data = {
                "entry_point": entry_point,
                "test": test_code,
                "timeout_seconds": timeout_seconds,
            }
            (sandbox_dir / "case.json").write_text(
                json.dumps(case_data, ensure_ascii=False), encoding="utf-8"
            )
            runner_path = sandbox_dir / "runner.py"
            runner_path.write_text(RUNNER_SOURCE, encoding="utf-8")

            # O processo dentro do container roda como usuário sem privilégios.
            # Permissões de leitura são necessárias para acessar o bind mount.
            sandbox_dir.chmod(0o755)
            for mounted_file in sandbox_dir.iterdir():
                mounted_file.chmod(0o644)

            mount_arg = (
                f"type=bind,source={sandbox_dir.resolve()},"
                "target=/work,readonly"
            )

            command = [
                "docker", "run", "--rm",
                "--network", "none",
                "--read-only",
                "--memory", "256m",
                "--memory-swap", "256m",
                "--cpus", "0.5",
                "--pids-limit", "32",
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges",
                "--user", "65534:65534",
                "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16m",
                "--mount", mount_arg,
                "--workdir", "/work",
                "--env", "PYTHONDONTWRITEBYTECODE=1",
                image,
                "python", "-I", "-B", "/work/runner.py",
            ]

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=OUTER_TIMEOUT,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return {
                    "status": "container_timeout",
                    "error_type": "ContainerTimeout",
                    "error_message": (
                        f"Container excedeu o limite externo de "
                        f"{OUTER_TIMEOUT:.0f} segundos."
                    ),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }

            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()

            for line in reversed(stdout.splitlines()):
                try:
                    payload = json.loads(line)
                    if isinstance(payload, dict) and "status" in payload:
                        return {
                            "status": payload.get("status", "runner_error"),
                            "error_type": payload.get("error_type", ""),
                            "error_message": payload.get("error_message", ""),
                            "elapsed_seconds": round(
                                time.monotonic() - started, 3
                            ),
                            "docker_exit_code": result.returncode,
                        }
                except json.JSONDecodeError:
                    continue

            detail = stderr[-1000:] if stderr else stdout[-1000:]
            return {
                "status": "docker_error",
                "error_type": "DockerExecutionError",
                "error_message": (
                    detail or f"Container terminou com código {result.returncode}."
                ),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "docker_exit_code": result.returncode,
            }

    except OSError as exc:
        return {
            "status": "runner_error",
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:1000],
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Executa os testes funcionais oficiais do HumanEval em Docker."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Pasta das amostras.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="HumanEval.jsonl.gz.")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Imagem Docker Python.")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Executa apenas as primeiras N amostras (0 = todas). Útil para teste inicial.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Limite interno, em segundos, para cada amostra.",
    )
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    sample_root = Path(args.root).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()

    if not sample_root.is_dir():
        print(f"Erro: pasta de amostras não encontrada: {sample_root}")
        return 1
    if args.timeout <= 0:
        print("Erro: --timeout precisa ser maior que zero.")
        return 1

    try:
        problems = load_humaneval(data_path)
    except (OSError, ValueError, FileNotFoundError) as exc:
        print(f"Erro ao carregar o HumanEval: {exc}")
        return 1

    ready, message = docker_ready()
    if not ready:
        print(f"Erro: {message}")
        return 1

    if not image_exists(args.image):
        print(f"Imagem Docker {args.image} não está disponível.")
        print(f"Execute primeiro: docker pull {args.image}")
        return 1

    source_files = sorted(sample_root.glob("HumanEval_*/sample_*.py"))
    if args.limit < 0:
        print("Erro: --limit não pode ser negativo.")
        return 1
    if args.limit > 0:
        source_files = source_files[:args.limit]
    if not source_files:
        print(f"Nenhum sample_*.py encontrado em {sample_root}")
        return 1

    results_dir = sample_root / "resultados_testes"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total = len(source_files)

    print("=== Testes funcionais HumanEval ===")
    print(f"Diretório: {sample_root}")
    print(f"Dataset: {data_path}")
    print(f"Imagem: {args.image}")
    print(f"Amostras encontradas: {total}")
    print(f"Timeout por amostra: {args.timeout:.1f}s")
    print("Docker: sem rede, filesystem somente leitura e limites ativos.\n")

    for index, source_path in enumerate(source_files, start=1):
        try:
            problem_number = int(source_path.parent.name.replace("HumanEval_", ""))
            sample_number = int(source_path.stem.replace("sample_", ""))
        except ValueError:
            continue

        task_id = f"HumanEval/{problem_number}"
        problem = problems.get(task_id)

        if problem is None:
            outcome = {
                "status": "missing_problem",
                "error_type": "DatasetError",
                "error_message": f"{task_id} não encontrado no dataset.",
                "elapsed_seconds": 0.0,
            }
        else:
            outcome = run_one(
                source_path=source_path,
                problem=problem,
                image=args.image,
                timeout_seconds=args.timeout,
                project_root=project_root,
            )

        status = outcome["status"]
        elapsed = outcome.get("elapsed_seconds", 0.0)
        print(
            f"[{index}/{total}] {task_id} amostra {sample_number}: "
            f"{status} ({elapsed:.2f}s)"
        )

        try:
            display_path = str(source_path.relative_to(project_root))
        except ValueError:
            display_path = str(source_path)

        rows.append(
            {
                "task_id": task_id,
                "sample": sample_number,
                "source_file": display_path,
                "status": status,
                "passed": status == "passed",
                "elapsed_seconds": elapsed,
                "error_type": outcome.get("error_type", ""),
                "error_message": outcome.get("error_message", ""),
                "docker_exit_code": outcome.get("docker_exit_code", ""),
                "model": "qwen2.5-coder:7b",
                "docker_image": args.image,
                "timeout_seconds": args.timeout,
            }
        )

    csv_path = results_dir / "resultados_humaneval.csv"
    json_path = results_dir / "resultados_humaneval.json"
    summary_path = results_dir / "resumo_humaneval.json"

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    passed = counts.get("passed", 0)
    failed = counts.get("failed", 0)
    timed_out = counts.get("timeout", 0) + counts.get("container_timeout", 0)
    not_evaluated = total - passed - failed - timed_out

    summary = {
        "total_samples_found": total,
        "passed": passed,
        "failed": failed,
        "timed_out": timed_out,
        "other_or_not_evaluated": not_evaluated,
        "status_counts": counts,
        "pass_rate_percent_over_all_found": round(
            passed / total * 100, 2
        ) if total else 0.0,
        "dataset": str(data_path),
        "docker_image": args.image,
        "timeout_seconds_per_sample": args.timeout,
        "execution": (
            "One disposable Docker container per sample; network disabled; "
            "read-only root filesystem; memory, CPU, process and time limits."
        ),
        "warning": (
            "Revise erros de Docker e casos não avaliados antes de calcular "
            "métricas. Validade sintática não comprova correção funcional."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== RESUMO ===")
    print(f"Total de amostras: {total}")
    print(f"Passaram: {passed}")
    print(f"Falharam nos testes: {failed}")
    print(f"Timeouts: {timed_out}")
    print(f"Outros/não avaliados: {not_evaluated}")
    rate = summary["pass_rate_percent_over_all_found"]
    print(f"Taxa de aprovação sobre todas as amostras: {rate:.2f}%")
    print("\nRelatórios salvos em:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {summary_path}")

    if any(
        row["status"] not in {"passed", "failed", "timeout"} for row in rows
    ):
        print(
            "\nAtenção: existem erros de infraestrutura ou casos não avaliados. "
            "Revise-os antes de calcular as métricas finais."
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
