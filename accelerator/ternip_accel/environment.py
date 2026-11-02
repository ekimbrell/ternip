"""Record build prerequisites; passing this audit does not qualify an image."""
import hashlib
import json
import platform
from pathlib import Path
import re
import shutil
import subprocess

from .platform import inspect_platform
from .target import validate


def find_tool(name, root=None):
    found = shutil.which(name)
    if found:
        return found
    if root:
        candidates = list(Path(root).glob(f'*/2022.2/bin/{name}*'))
        candidates += list(Path(root).glob(f'*/2023.2/bin/{name}*'))
        for path in candidates:
            if path.name in (name, name + '.bat', name + '.exe') and path.is_file():
                return str(path.resolve())
    return None


def audit(target, output, tool_root=None, platform_file=None):
    target = validate(target)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = dict(status='blocked', hardware_qualified=False, target=target,
                  host_system=platform.system(), host_release=platform.release(), tools={}, blockers=[])
    blockers = record['blockers']
    if record['host_system'] != 'Linux':
        blockers.append('Use a Linux build environment for the pinned Alveo/XRT flow')

    def run(label, command):
        try:
            result = subprocess.run(command, cwd=out, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, timeout=180)
            text, code = result.stdout, result.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            text, code = str(exc), -1
        (out / (label + '.log')).write_text(text, encoding='utf-8')
        return code, text

    for name, flag, required in [('vivado', '-version', '2022.2'),
                                  ('v++', '--version', '2022.2'),
                                  ('xbutil', '--version', '2.14.354')]:
        path = find_tool(name, tool_root)
        info = record['tools'][name] = dict(path=path, required_version=required)
        if not path:
            blockers.append(f'Missing {name}')
            continue
        code, text = run(name.replace('+', 'p') + '-version', [path, flag])
        info['version_output_sha256'] = hashlib.sha256(text.encode()).hexdigest()
        info['version_output'] = text
        info['version_matches'] = code == 0 and bool(re.search(r'(?<![\d.])' + re.escape(required) + r'(?![\d.])', text))
        if not info['version_matches']:
            blockers.append(f'{name} does not report required version {required}')

    vivado = record['tools']['vivado']
    if vivado.get('version_matches'):
        # The part comes from the validated descriptor, never arbitrary Tcl.
        part = target['part']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', part):
            raise ValueError('Unsafe part identifier')
        script = out / 'part.tcl'
        script.write_text(f'if {{[llength [get_parts -quiet {part}]] != 1}} {{exit 1}}\nputs "TERNIP_PART_PRESENT"\nexit 0\n')
        code, text = run('part', [vivado['path'], '-mode', 'batch', '-notrace', '-nojournal', '-source', str(script)])
        record['part_installed'] = code == 0 and 'TERNIP_PART_PRESENT' in text.splitlines()
        if not record['part_installed']:
            blockers.append(f'Required part {part} unavailable')

    platform_tool = find_tool('platforminfo', tool_root)
    record['tools']['platforminfo'] = dict(path=platform_tool)
    if not platform_tool:
        blockers.append('Missing platforminfo')
    if platform_file is None:
        blockers.append('Supply the installed development platform .xpfm with --platform-file')
    else:
        path = Path(platform_file).resolve()
        if not path.is_file() or path.suffix != '.xpfm' or path.stem != target['platform']:
            blockers.append('Platform file is missing or does not match the target platform identifier')
        else:
            record['platform_file'] = str(path)
            record['platform_file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            if platform_tool:
                code, text = run('platforminfo', [platform_tool, '-p', str(path), '--json'])
                try:
                    if code:
                        raise ValueError('platforminfo failed')
                    record['platform_contract'] = inspect_platform(json.loads(text), target)
                    record['platforminfo_sha256'] = hashlib.sha256(text.encode()).hexdigest()
                except (ValueError, TypeError) as exc:
                    blockers.append(f'Platform evidence rejected: {exc}')
    if not blockers:
        record['status'] = 'preflight-passed-not-qualified'
    (out / 'environment.json').write_text(json.dumps(record, indent=2) + '\n')
    return record
