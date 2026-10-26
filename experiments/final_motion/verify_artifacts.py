"""Проверка готовых файлов отчёта и фиксация контрольных сумм поставки."""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

from .train import OUT, ROOT


def main():
    checked_links = 0
    documents = [ROOT / "README.md", ROOT / "research/motion_model_report.md",
                 ROOT / "experiments/motion_pilot_float_timing/README.md"]
    for path in documents:
        content = path.read_text(encoding="utf-8")
        assert not any(ord(c) < 32 and c not in "\n\t" for c in content)
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if "://" in target or target.startswith("#"):
                continue
            target = target.split("#")[0].strip("<>")
            assert (path.parent / unquote(target)).resolve().exists(), (path, target)
            checked_links += 1
    report = documents[1].read_text(encoding="utf-8")
    assert report.count("$$") == 4 and report.count(chr(96) * 3) % 2 == 0
    assert "\\sum_{p}" in report and "\\lambda_k" in report and "\\operatorname{med}" in report
    for phrase in ["Важно отметить", "Стоит подчеркнуть", "Необходимо понимать", "Играет важную роль",
                   "Таким образом, можно сделать вывод", "Это не просто"]:
        assert phrase not in report, phrase
    for source in (ROOT / "experiments/final_motion").glob("*.py"):
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    plots = sorted((ROOT / "research/figures").glob("motion_*.png"))
    assert len(plots) == 7 and all(p.stat().st_size > 5000 for p in plots)
    weights = sorted((OUT / "checkpoints").glob("*.pt"))
    assert len(weights) == 136
    for weight in [OUT / "deployment.pt", weights[0]]:
        ignored = subprocess.run(["git", "check-ignore", "--quiet", str(weight)], cwd=ROOT)
        assert ignored.returncode == 1, f"Веса исключены из Git: {weight}"
    cache = subprocess.run(["git", "check-ignore", "--quiet", str(OUT / "amplitudes.npy")], cwd=ROOT)
    assert cache.returncode == 0
    paths = list((ROOT / "experiments/final_motion").glob("*.py"))
    paths += [p for p in OUT.rglob("*") if p.is_file() and p.name not in ("amplitudes.npy", "artifact_manifest.json")]
    paths += [p for p in (ROOT / "experiments/motion_pilot_float_timing").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    paths += plots + documents + [ROOT / ".gitignore"]
    paths = sorted(set(paths))
    files = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                  sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    result = dict(state="passed", files=len(files), bytes_without_raw_cache=sum(r["bytes"] for r in files),
                  local_links_checked=checked_links, plots=len(plots), checkpoints=len(weights),
                  final_weights_included_by_gitignore=True, raw_cache_ignored=True, entries=files)
    (OUT / "artifact_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in result.items() if k != "entries"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
