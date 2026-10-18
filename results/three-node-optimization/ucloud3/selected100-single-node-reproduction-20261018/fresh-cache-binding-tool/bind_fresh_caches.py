#!/usr/bin/env python3
"""Bind fresh fitted caches to an isolated unmeasured portable deployment (CPU only)."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, re, shutil, time
from pathlib import Path

TEMPLATE_RECIPE_SHA = '8cb2bec6883efaeec7b69edf89acd27205201c2773b4fe175195f31c93a94bd7'
TEMPLATE_ASSET_SHA = '7f1f5dbd1b6b288f165852772ed0c8090cc8f5abc0a73265a14f3b604d0b911a'
TEMPLATE_FULL_SHA = 'b7a95de6dedd2319e9e1c7b32e21775e7d1b7fb0bee84e6f3ded216cf25abc1b'
RUNNER_SHA = 'd80df390cf6d1d50129796b31fdd7f4d1e95d0043d3b942d47ac582118702fb0'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    require(path.is_file(), f'Not a file: {path}')
    result = dict(bytes=before.st_size, sha256=digest(path))
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f'Input changed while hashing: {path}')
    return result

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')

def cache_inventory(root, selected, full, allow_full):
    """Validate exact serving shape, hash only immutable selected serving files."""
    root = Path(root).resolve(strict=True)
    require(root.is_dir(), f'Cache root is not a directory: {root}')
    selected_dirs = {str(Path(p).parent) for p in selected['files']}
    full_dirs = {str(Path(p).parent) for p in full['files']}
    actual_dirs = {p.name for p in root.glob('weights-L*')}
    allowed = [selected_dirs] + ([full_dirs] if allow_full else [])
    require(actual_dirs in allowed, f'Unknown/missing serving layer directories: {root}')
    all_spec = full if actual_dirs == full_dirs else selected
    actual_files = set()
    for directory in actual_dirs:
        dp = root / directory
        require(dp.is_dir(), f'Not a serving directory: {dp}')
        # Fitter auxiliaries are permitted but never copied into the serving view.
        names = {p.name for p in dp.iterdir()}
        serving = {'packed.bin', 'scales.f32'}
        auxiliaries = {'expanded.i8', 'fit-equivalence.json', 'hessians.json', 'fit-frozen-before-heldout.json'}
        require(serving <= names and names <= serving | auxiliaries, f'Unknown/missing serving file: {dp}: {names}')
        for name in serving:
            rel = directory + '/' + name
            require((root / rel).is_file(), f'Not a regular payload file: {root / rel}')
            require((root / rel).stat().st_size == all_spec['files'][rel]['bytes'],
                    f'Payload size/format mismatch: {root / rel}')
            actual_files.add(rel)
    require(actual_files == set(all_spec['files']), f'Unexpected serving inventory: {root}')
    new = copy.deepcopy(selected)
    new['files'] = {rel: identity(root / rel) for rel in sorted(selected['files'])}
    new['file_count'] = len(new['files'])
    new['total_bytes'] = sum(v['bytes'] for v in new['files'].values())
    return new, dict(source=str(root), source_serving_files=len(actual_files),
                     selected_serving_files=len(new['files']), omitted_serving_files=len(actual_files)-len(new['files']))

def fresh_recipe(template, recipe_id, binary_id, source_sha, contract_sha, origins, evidence):
    require(re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}', recipe_id) is not None, 'Unsafe recipe ID')
    require(re.fullmatch(r'[0-9a-f]{64}', source_sha) is not None, 'Source SHA must be a 64-character lowercase hex identity')
    recipe = copy.deepcopy(template)
    recipe.update(recipe_id=recipe_id, version_id=recipe_id, binary_artifact_id=recipe_id,
                  binary_sha256=binary_id['sha256'], source_sha256=source_sha,
                  selection_status='fresh_cache_unmeasured_0_performance_requests',
                  quality_status='fresh_cache_unmeasured_0_quality_questions',
                  qualification='Fresh cache content identity only. No inherited performance or quality. Numerical calibration/fit/native qualification is separate and required before evaluation.')
    recipe['assets'] = dict(contract=dict(file='assets-u3-47.json', sha256=contract_sha),
                            original_cache_roots={k:v['source'] for k,v in origins.items()},
                            verification='All selected 478 packed/scales payloads freshly SHA256 hashed. Original 47-layer mask, file format, model identity and 19 environment values preserved. SHA is identity, not numerical qualification.',
                            binding_evidence=dict(file='evidence/fresh-binding.json', sha256=evidence))
    # Do not copy historical cache/model-equivalence receipts or measured statuses.
    for key in ('rebuild_provenance', 'measured_results', 'performance_results', 'quality_results'):
        recipe.pop(key, None)
    return recipe

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template-package', required=True, type=Path)
    for name in ('weights025','weights041','weights047','binary','model','tokenizer','build-env','eval-envs','tools','modelscope-cache','fit-evidence'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--recipe-id', required=True)
    p.add_argument('--output', required=True, type=Path, help='New deployment root; must not exist')
    a = p.parse_args(argv)
    out = a.output.absolute()
    require(not out.exists() and not out.is_symlink(), 'Output must be new; never overwrite an existing deployment')
    package = a.template_package.resolve(strict=True)
    catalog = package / 'candidate_recipes'
    for name, expected in [('100-k3-u3-47.json', TEMPLATE_RECIPE_SHA), ('assets-u3-47.json', TEMPLATE_ASSET_SHA), ('assets-full64.json', TEMPLATE_FULL_SHA)]:
        require(digest(catalog/name) == expected, f'Original fixed template mismatch: {name}')
    require(digest(package/'organizer/runner.py') == RUNNER_SHA, 'Official runner SHA mismatch')
    require('qualification=recipe.get("qualification",' in (package/'candidate_common.py').read_text(), 'Template must support explicit unmeasured qualification metadata')
    template = json.loads((catalog/'100-k3-u3-47.json').read_text())
    contract = json.loads((catalog/'assets-u3-47.json').read_text())
    full = json.loads((catalog/'assets-full64.json').read_text())
    require(len(template['environment']) == 19, 'Expected exact original 19 environment values')
    require(re.fullmatch(r'[0-9a-f]{64}', a.source_sha256) is not None, 'Invalid source identity')
    require(re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}', a.recipe_id) is not None, 'Unsafe recipe ID')
    origins = {}
    for name in ('weights025', 'weights041', 'weights047'):
        contract['caches'][name], origins[name] = cache_inventory(getattr(a,name), contract['caches'][name], full['caches'][name], name=='weights047')
    require(sum(x['file_count'] for x in contract['caches'].values()) == 478, 'Selected mask must contain exactly 478 payload files')
    actual_model = identity(a.model)
    require(actual_model == contract['model'], 'Fixed original model content identity mismatch')
    actual_binary = identity(a.binary)
    require(os.access(a.binary, os.X_OK), 'Binary must be executable')
    fit = identity(a.fit_evidence)
    links = {'assets/model/qwen3_8_27b.ninfer': a.model, 'assets/tokenizer': a.tokenizer,
             'envs/build': a.build_env, 'eval-envs': a.eval_envs, 'tools': a.tools, 'cache/modelscope': a.modelscope_cache,
             'artifacts/'+a.recipe_id+'/ninfer-serve': a.binary}
    for rel, source in links.items():
        require(source.exists(), f'Missing deployment dependency: {source}')
    original_recipe_digest = digest(catalog/'100-k3-u3-47.json')
    out.mkdir(parents=True)
    entry = out/'entrypoint'
    # Copy executable runner, cohort, salt, readiness helper; omit old reports/recipes.
    shutil.copytree(package, entry, ignore=shutil.ignore_patterns('candidate_recipes', 'evidence', 'checks', 'tests', 'verify_cpu.py', 'README.md', 'MANIFEST.json', '__pycache__', '*.pyc'))
    ec = entry/'candidate_recipes'; ec.mkdir()
    write_json(ec/'assets-u3-47.json', contract)
    for rel, source in links.items():
        dest = out/rel; dest.parent.mkdir(parents=True, exist_ok=True); dest.symlink_to(source.resolve(strict=True))
    for name, spec in contract['caches'].items():
        for rel in spec['files']:
            dest = out/'assets'/name/rel; dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to((getattr(a,name)/rel).resolve(strict=True))
    (entry/'evidence').mkdir()
    shutil.copy2(a.fit_evidence, entry/'evidence/fit-evidence.json')
    binding = dict(schema_version=1, created_ns=time.time_ns(), recipe_id=a.recipe_id,
                   source_sha256=a.source_sha256, binary=actual_binary, binary_path=str(a.binary.resolve()),
                   model=actual_model, origins=origins, selected_cache_files=478,
                   fixed_template=dict(recipe_sha256=TEMPLATE_RECIPE_SHA, asset_sha256=TEMPLATE_ASSET_SHA, full_asset_sha256=TEMPLATE_FULL_SHA, runner_sha256=RUNNER_SHA),
                   fit_evidence=dict(**fit, source=str(a.fit_evidence.resolve()), interpretation='Identity/provenance only; this binder does not assert numerical qualification.'),
                   environment=template['environment'], performance_requests=0, quality_questions=0,
                   scope='CPU content-hash binding and deployment symlinks only; no GPU, HTTP, build or evaluation performed.')
    write_json(entry/'evidence/fresh-binding.json', binding)
    recipe = fresh_recipe(template, a.recipe_id, actual_binary, a.source_sha256, digest(ec/'assets-u3-47.json'), origins, digest(entry/'evidence/fresh-binding.json'))
    write_json(ec/(a.recipe_id+'.json'), recipe)
    write_json(ec/'index.json', dict(schema_version=1, recipes={a.recipe_id:digest(ec/(a.recipe_id+'.json'))}, scope='Fresh fitted cache candidate; unmeasured. No inherited frozen100 scores.'))
    write_json(entry/'evidence/status.json', dict(recipe_id=a.recipe_id, performance_requests=0, quality_questions=0, numerical_qualification='not asserted by binder'))
    write_json(out/'deployment-links.json', {k:str(v.resolve()) for k,v in links.items()})
    files = {str(f.relative_to(entry)):identity(f) for f in sorted(entry.rglob('*')) if f.is_file()}
    write_json(entry/'MANIFEST.json', dict(schema_version=1, scope='Fresh-cache independent package', files=files))
    require(digest(catalog/'100-k3-u3-47.json') == original_recipe_digest, 'Original template changed unexpectedly')
    print(json.dumps(dict(status='BOUND_UNMEASURED', root=str(out), recipe=str(ec/(a.recipe_id+'.json')), recipe_sha256=digest(ec/(a.recipe_id+'.json')), asset_sha256=digest(ec/'assets-u3-47.json'), binary=actual_binary, selected_cache_files=478, numerical_qualification='caller must validate fit/native gates'), indent=2))

if __name__ == '__main__':
    main()
