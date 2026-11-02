import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ternip_accel import environment

TARGET = json.loads((Path(__file__).parents[1] / 'targets/u280-ddr-rtl.json').read_text())


def setup(monkeypatch, tmp_path, wrong_version=False, part_present=True):
    monkeypatch.setattr(environment.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(environment, 'find_tool', lambda name, root=None: name)
    platform = tmp_path / (TARGET['platform'] + '.xpfm')
    platform.write_text('test fixture, not a vendor platform')
    def run(command, **kwargs):
        if command[0] == 'platforminfo':
            output = json.dumps(dict(part=TARGET['part'], numSlrs=3,
                memories=['DDR[0]', 'DDR[1]'], clockFrequencyMHz=300))
        elif '-source' in command:
            output = 'TERNIP_PART_PRESENT\n' if part_present else ''
        else:
            output = '2.14.354' if command[0] == 'xbutil' else ('2023.2' if wrong_version else '2022.2')
        return SimpleNamespace(returncode=0, stdout=output)
    monkeypatch.setattr(environment.subprocess, 'run', run)
    return platform


def test_complete_evidence_is_recorded_not_qualified(monkeypatch, tmp_path):
    platform = setup(monkeypatch, tmp_path)
    report = environment.audit(TARGET, tmp_path/'audit', platform_file=platform)
    assert report['status'] == 'preflight-passed-not-qualified'
    assert not report['hardware_qualified']
    assert report['part_installed']
    assert len(report['platform_file_sha256']) == 64
    assert report['platform_contract']['slrs'] == [0, 1, 2]


@pytest.mark.parametrize('failure', ['version', 'part', 'platform'])
def test_partial_installation_never_passes(monkeypatch, tmp_path, failure):
    platform = setup(monkeypatch, tmp_path, wrong_version=failure=='version', part_present=failure!='part')
    result = environment.audit(TARGET, tmp_path/'audit', platform_file=None if failure=='platform' else platform)
    assert result['status'] == 'blocked'
    assert result['blockers']


def test_missing_tools_report_all_blockers(monkeypatch, tmp_path):
    monkeypatch.setattr(environment, 'find_tool', lambda *args: None)
    result = environment.audit(TARGET, tmp_path/'audit')
    for name in ['vivado', 'v++', 'xbutil', 'platforminfo']:
        assert 'Missing '+name in result['blockers']
