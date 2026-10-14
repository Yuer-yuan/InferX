"""Deterministic repository task preparation; never execute downloaded setup scripts."""
import ast
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile


def parse_completion(text: str) -> str:
    text = text.strip()
    fence = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if fence:
        text = fence.group(1)
    value = json.loads(text)
    if not isinstance(value, dict) or not isinstance(value.get('code'), str):
        raise ValueError('completion must be a JSON object with string code')
    return value['code']


def safe_relative(repo: str, filename: str) -> Path:
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', repo) or repo in ('.', '..'):
        raise ValueError('invalid repository name')
    parts = filename.removeprefix('/').split('/')
    if (len(parts) < 2 or parts[0] != repo or '\\' in filename
            or any(p in ('', '.', '..') for p in parts)):
        raise ValueError('file must be inside the named repository')
    return Path(*parts[1:])


def extract_repositories(archive: Path, destination: Path, repositories: list[str]) -> None:
    ignored = {'.git', '.pytest_cache', '__pycache__', '.benchmarks', '.hypothesis',
               '.mypy_cache', 'build', 'dist', '.tox', '.venv'}
    with zipfile.ZipFile(archive) as z:
        chosen = []
        for item in z.infolist():
            parts = item.filename.rstrip('/').split('/')
            if len(parts) < 2 or parts[0] != 'repos' or parts[1] not in repositories:
                continue
            if any(p in ('', '.', '..') for p in parts) or '\\' in item.filename:
                raise ValueError('unsafe archive path')
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('archive symlink refused')
            if any(p in ignored for p in parts[2:]) or item.filename.endswith(('.pyc', '.pyo')):
                continue
            chosen.append(item)
        destination.mkdir(parents=True, exist_ok=False)
        for item in chosen:
            path = destination.joinpath(*PurePosixPath(item.filename).parts[1:])
            if item.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(item) as source, path.open('wb') as target:
                    shutil.copyfileobj(source, target)


def ablated_source(row: dict) -> str:
    prefix, suffix = row['prefix_code'], row['suffix_code']
    if prefix.rsplit('\n', 1)[-1].strip() or (suffix and not suffix.startswith('\n')):
        raise ValueError('masked span does not have a whole-block boundary')
    text = prefix + 'pass' + suffix
    try:
        ast.parse(text)
    except SyntaxError as exc:
        raise ValueError('pass ablation is not syntactically valid') from exc
    return text


def reference_matches(row: dict, fixture_root: Path) -> bool:
    path = fixture_root/row['repo_name']/safe_relative(row['repo_name'], row['file_name'])
    return path.read_text() == row['prefix_code'] + row['middle_code'] + row['suffix_code']


def line_start_task(row: dict) -> dict:
    """Move known prefix indentation into the requested gap, before generation."""
    indent = row['prefix_code'].rsplit('\n', 1)[-1]
    if indent.strip():
        raise ValueError('only whole-block boundaries can move to line start')
    return dict(row, prefix_code=row['prefix_code'][:-len(indent)] if indent else row['prefix_code'],
                middle_code=indent + row['middle_code'], infill_indent=indent)


def source_context(row: dict, limit: int = 6000) -> str:
    result = []
    remaining = limit
    for name, code in row['context_code']:
        parts = PurePosixPath(name).parts
        if (name.lstrip('/') == row['file_name'].lstrip('/') or not name.endswith('.py')
                or any(p.startswith('test') or p == 'conftest.py' for p in parts)):
            continue
        entry = f'File: {name}\n{code}'
        if len(entry) > remaining:
            entry = entry[:remaining].rsplit('\n', 1)[0]
        if entry:
            result.append(entry)
            remaining -= len(entry) + 2
        if remaining <= 0:
            break
    return '\n\n'.join(result)
