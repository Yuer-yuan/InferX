"""Extract the public QoServe policy without its legacy CUDA/runtime imports.

This is a compatibility port, not a new policy. The generated MIT module
retains the pinned author's numerical branches; native request/cache ownership
belongs to the later adapter. Reference A100 coefficients are never deployment
defaults. Supplying coefficients here does not certify a calibrated profile.
"""
import ast
import copy
import hashlib
import io
import math
from pathlib import Path
import tarfile


AUTHOR_COMMIT = '72339db5b120614033b317880ad3a05a5e4c9415'
CAPSULE_SHA256 = 'dfadf6fde36bf3671f4f2dbe4c354f595d9d4ad1f851bac9e9d29a7e10ae02c0'
TIMING_SHA256 = '0ea48ab80e4236feb1d5944a30bb5acf2b165bd6b27843c881027daf9ff5e9e1'
LICENSE_SHA256 = 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'
SCHEDULER = 'sarathi/core/scheduler/deadline_scheduler.py'
SEQUENCE = 'sarathi/core/datatypes/sequence.py'
CONFIG = 'sarathi/config/config.py'
FEATURES = ('batch_num_tokens', 'batch_num_prefill_tokens', 'batch_num_decode_tokens',
            'batch_decode_context', 'batch_prefill_context')
BINS = ('tokens_0_512', 'tokens_513_plus')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _verified_sources(capsule, manifest):
    raw = Path(capsule).read_bytes()
    if _sha(raw) != CAPSULE_SHA256 or manifest.get('capsule_sha256') != CAPSULE_SHA256:
        raise ValueError('author capsule hash mismatch')
    if manifest.get('commit') != AUTHOR_COMMIT:
        raise ValueError('author commit mismatch')
    if (manifest.get('repo') != 'https://github.com/microsoft/sarathi-serve'
            or manifest.get('branch') != 'niyama_asplos2026'):
        raise ValueError('author repository identity mismatch')
    # Read bytes in memory; never extract paths or follow links from an archive.
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        members = archive.getmembers()
        if any(not member.isfile() for member in members):
            raise ValueError('source capsule contains a non-file member')
        if len({member.name for member in members}) != len(members):
            raise ValueError('source capsule has duplicate members')
        sources = {member.name: archive.extractfile(member).read() for member in members}
    files = manifest.get('files')
    if not isinstance(files, dict) or set(files) != set(sources):
        raise ValueError('author source inventory mismatch')
    for name, value in sources.items():
        if files[name] != _sha(value):
            raise ValueError(f'author source hash mismatch: {name}')
    if files.get('LICENSE') != LICENSE_SHA256:
        raise ValueError('author license hash mismatch')
    return {name: raw.decode() for name, raw in sources.items()}


def _class(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _method(cls, name):
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _assigns_name(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                               for target in node.targets)


def _assigns_attr(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name) and target.value.id == 'self'
        and target.attr == name for target in node.targets)


def _validate_coefficients(coefficients):
    if not isinstance(coefficients, dict) or set(coefficients) != set(BINS):
        raise ValueError('deployment coefficients need both exact predictor bins')
    for name in BINS:
        model = coefficients[name]
        if (not isinstance(model, dict) or set(model) != {'coefficients', 'intercept'}
                or not isinstance(model['coefficients'], dict)
                or set(model['coefficients']) != set(FEATURES)):
            raise ValueError('deployment coefficients have invalid feature tables')
        for value in (model['intercept'], *model['coefficients'].values()):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('deployment coefficients must be finite numbers')
    return copy.deepcopy(coefficients)


class _NativeCapacity(ast.NodeTransformer):
    """Replace old allocator access with the adapter's truthful capacity query."""
    def visit_Call(self, node):
        if ast.unparse(node) == 'self.block_manager.gpu_allocator.get_num_free_blocks()':
            return ast.copy_location(ast.parse('self._qoserve_effective_free_blocks()').body[0].value, node)
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        if ast.unparse(node) == 'self.block_manager.block_size':
            return ast.copy_location(ast.parse('self.block_size').body[0].value, node)
        return self.generic_visit(node)


def _function(name, arguments, body):
    node = ast.parse(f'def {name}({arguments}): pass').body[0]
    node.body = copy.deepcopy(body)
    return node


def _build(capsule, manifest, coefficients):
    sources = _verified_sources(capsule, manifest)
    tree = ast.parse(sources[SCHEDULER])
    scheduler = _class(tree, 'DeadlineScheduler')
    sequence = _class(ast.parse(sources[SEQUENCE]), 'Sequence')
    config = _class(ast.parse(sources[CONFIG]), 'DeadlineSchedulerConfig')
    methods, originals = {}, {}

    for name in ('_predict_batch_time', '_get_prefill_size_by_slack', '_sorting_key'):
        original = _method(scheduler, name)
        originals[name] = ast.get_source_segment(sources[SCHEDULER], original)
        methods[name] = copy.deepcopy(original)
    reference = {}
    predictor = methods['_predict_batch_time']
    for index, node in enumerate(predictor.body):
        for bin_name in BINS:
            if _assigns_name(node, bin_name):
                reference[bin_name] = ast.literal_eval(node.value)
                node.value = ast.parse(f'self._qoserve_coefficients[{bin_name!r}]').body[0].value
    selected = _validate_coefficients(reference if coefficients is None else coefficients)
    methods['_get_prefill_size_by_slack'] = _NativeCapacity().visit(methods['_get_prefill_size_by_slack'])

    for name in ('__lt__', '_get_TBT_deadline'):
        original = _method(sequence, name)
        originals[name] = ast.get_source_segment(sources[SEQUENCE], original)
        methods[name] = copy.deepcopy(original)

    schedule = _method(scheduler, '_schedule')
    tier = next(node for node in ast.walk(schedule) if isinstance(node, ast.If)
                and ast.unparse(node.test) == 'seq.tier >= 1')
    originals['_qoserve_ttft_budget'] = ast.get_source_segment(sources[SCHEDULER], tier)
    methods['_qoserve_ttft_budget'] = _function('_qoserve_ttft_budget', 'self, seq',
        [tier, ast.parse('return seq.ttft_deadline').body[0]])

    prefill_loop = next(node for node in schedule.body if isinstance(node, ast.While)
                        and ast.unparse(node.test) == 'self._prefill_queue')
    remaining = next(node for node in prefill_loop.body if _assigns_name(node, 'remaining_prefill_tokens'))
    mode_guard = next(node for node in prefill_loop.body if isinstance(node, ast.If)
                      and "self.scheduler_type == 'deadline'" in ast.unparse(node.test))
    originals['_qoserve_relegate_once'] = '\n'.join(
        ast.get_source_segment(sources[SCHEDULER], node) for node in (remaining, mode_guard))
    relegate = _function('_qoserve_relegate_once', 'self, seq, now', [remaining, mode_guard])
    # Only the original inner continue becomes this helper's True return.
    inner = next(node for node in ast.walk(relegate) if isinstance(node, ast.If)
                 and 'seq.drop == 0' in ast.unparse(node.test))
    if not isinstance(inner.body[-1], ast.Continue):
        raise ValueError('author relegation boundary mismatch')
    inner.body[-1] = ast.parse('return True').body[0]
    relegate.body.append(ast.parse('return False').body[0])
    methods[relegate.name] = relegate

    ema_index = next(index for index, node in enumerate(schedule.body) if isinstance(node, ast.If)
                     and ast.unparse(node.test) == 'self.last_prefill_size != 0')
    ema = schedule.body[ema_index:ema_index + 3]
    if not (_assigns_attr(ema[1], 'last_prefill_size') and _assigns_attr(ema[2], 'prev_call_time')):
        raise ValueError('author throughput observation boundary mismatch')
    originals['_qoserve_update_prefill_throughput'] = '\n'.join(
        ast.get_source_segment(sources[SCHEDULER], node) for node in ema)
    methods['_qoserve_update_prefill_throughput'] = _function('_qoserve_update_prefill_throughput', 'self, now', ema)

    defaults = {}
    for node in config.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call):
            default = next(keyword.value for keyword in node.value.keywords if keyword.arg == 'default')
            defaults[node.target.id] = ast.literal_eval(default)
    for node in _method(scheduler, '__init__').body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if isinstance(value, (int, float, str)):
                defaults[node.targets[0].attr] = value

    candidates = [copy.deepcopy(node) for node in tree.body
                  if _assigns_name(node, 'TOKEN_SEARCH_SPACE') or _assigns_name(node, 'LOW_MEMORY_TOKEN_SEARCH_SPACE')]
    sequence_cls = ast.ClassDef(name='AuthorSequence', bases=[], keywords=[], decorator_list=[],
                               body=[methods['__lt__'], methods['_get_TBT_deadline']])
    policy_body = [ast.parse(f'{name} = {value!r}').body[0] for name, value in defaults.items()]
    policy_body.extend(ast.parse(f'''Sequence = AuthorSequence
_qoserve_coefficients = {selected!r}
deployment_coefficients_supplied = {coefficients is not None!r}
coefficient_source = {'hardware-supplied' if coefficients is not None else 'author-reference'!r}
@classmethod
def require_deployment_coefficients(cls):
    if not cls.deployment_coefficients_supplied:
        raise ValueError('deployment requires hardware coefficients, not author A100 reference tables')
''').body)
    policy_body.extend(node for name, node in methods.items() if name not in ('__lt__', '_get_TBT_deadline'))
    policy_cls = ast.ClassDef(name='AuthorPolicy', bases=[], keywords=[], decorator_list=[], body=policy_body)
    module = ast.fix_missing_locations(ast.Module(body=candidates + [sequence_cls, policy_cls], type_ignores=[]))
    generated = ('# SPDX-License-Identifier: MIT\n# Copyright (c) Microsoft Corporation.\n'
                 f'# QoServe source commit {AUTHOR_COMMIT}; see bundled author LICENSE.\n'
                 'import heapq\n\n' + ast.unparse(module) + '\n')
    return generated, dict(
        commit=AUTHOR_COMMIT, capsule_sha256=CAPSULE_SHA256, license_sha256=LICENSE_SHA256,
        source_files=copy.deepcopy(manifest['files']),
        source_defaults=defaults, reference_coefficients=reference, coefficients=selected,
        coefficient_source='hardware-supplied' if coefficients is not None else 'author-reference',
        deployment_coefficients_supplied=coefficients is not None,
        generated_module_sha256=_sha(generated.encode()),
        methods={name: dict(source_sha256=_sha(originals[name].encode()),
                            generated_sha256=_sha(ast.unparse(ast.fix_missing_locations(node)).encode()))
                 for name, node in methods.items()},
        mappings=[
            'Only predictor coefficient table assignments are injected; numerical formula/bin edge unchanged.',
            'Old free-block/block-size reads map to native effective capacity and block_size.',
            'Original Sequence comparison and cumulative TBT methods are extracted unchanged.',
            'Tier budget and EMA statements are extracted into callable helpers.',
            'Eager-relegation requeue is unchanged; its inner continue maps to helper return True.',
            'Native dispatcher/whole-input commitments are separate compatibility code, not unchanged upstream.',
            'Supplied coefficients alone do not qualify predictor identity/coverage/holdout accuracy.',
        ])


def load_author_policy(capsule: Path, manifest: dict, coefficients: dict | None = None) -> type:
    """Return executable stdlib-only policy; no legacy imports or file writes."""
    source, _ = _build(capsule, manifest, coefficients)
    namespace = {'__name__': 'inferx_qoserve_author_policy'}
    exec(compile(source, '<pinned-qoserve-policy>', 'exec'), namespace)
    cls = namespace['AuthorPolicy']
    cls._module_source = source
    return cls


def policy_manifest(capsule: Path, manifest: dict, coefficients: dict | None = None,
                    *, timing_source: Path) -> dict:
    """Bind exact source, mapped method bodies, hardware tables and timing proof."""
    timing_bytes = Path(timing_source).read_bytes()
    if _sha(timing_bytes) != TIMING_SHA256:
        raise ValueError('author timing source hash mismatch')
    _, result = _build(capsule, manifest, coefficients)
    result['timing_source_sha256'] = TIMING_SHA256
    result['timing_boundary'] = 'schedule entry through native output/scheduler update exit'
    result['author_timing_source'] = 'sarathi/engine/base_llm_engine.py'
    return result
