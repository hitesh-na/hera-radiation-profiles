"""Install the reviewed local update, backing up known original code/config files."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install(root):
    root=Path(root).resolve()
    bundle=root/'updates/time_matching_20261014'
    manifest=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    if not (root/'pfotzer_xgb/__init__.py').is_file():
        raise ValueError('Extract into your existing Radiation_Profiles_Pfotzer_Peak_Estimation folder first.')
    dependency=root/'scripts/diagnose_real_data.py'
    if not dependency.is_file() or digest(dependency)!=manifest['diagnostics_dependency_sha256']:
        raise ValueError('The expected diagnostics add-on is missing or changed. Keep both earlier add-ons and share this message.')
    operations=[]
    for item in manifest['files']:
        rel=Path(item['path']);target=(root/rel).resolve();source=(bundle/'files'/rel).resolve()
        if (rel.is_absolute() or not target.is_relative_to(root) or not source.is_relative_to(bundle/'files') or
                not rel.parts or rel.parts[0] not in {'pfotzer_xgb','config','scripts','docs','tests','results'}):
            raise ValueError('Unexpected update path')
        if not source.is_file() or digest(source)!=item['sha256']:
            raise ValueError(f'Update file missing or damaged: {rel}')
        if target.exists():
            if not target.is_file():raise ValueError(f'Expected a file at {rel}')
            current=digest(target)
            if current==item['sha256']:continue
            if current!=item.get('original_sha256'):
                raise ValueError(f'{rel} differs from the supplied project. Nothing was changed. Share this message; do not force an overwrite.')
        elif item.get('original_sha256'):
            raise ValueError(f'Expected original project file is missing: {rel}')
        operations.append((source,target,rel))
    if not operations:
        print('This update is already installed. No changes needed.');return None
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=root/'results/code_update_backups'/stamp
    backup.mkdir(parents=True,exist_ok=False)
    journal=[]
    for source,target,rel in operations:
        original=target.exists()
        if original:
            saved=backup/rel;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,saved)
        journal.append({'path':rel.as_posix(),'had_original':original})
    (backup/'update_journal.json').write_text(json.dumps(journal,indent=2)+'\n',encoding='utf-8')
    completed=[]
    try:
        for source,target,rel in operations:
            target.parent.mkdir(parents=True,exist_ok=True)
            completed.append((target,rel))
            shutil.copy2(source,target)
            if digest(target)!=digest(source):raise OSError(f'Copy verification failed: {rel}')
    except Exception:
        for target,rel in reversed(completed):
            saved=backup/rel
            if saved.exists():shutil.copy2(saved,target)
            elif target.exists():target.unlink()
        raise
    print(f'Installed {len(operations)} reviewed code/config/report files.')
    print(f'Original code/config backup: {backup}')
    print('Raw recordings, notebooks and earlier results are unchanged.')
    print('Keep RUN_TRAINING=False. Next run: .\\.venv\\Scripts\\python.exe .\\scripts\\review_time_matching.py')
    return backup


if __name__=='__main__':
    try:install(Path(__file__).resolve().parents[1])
    except (OSError,ValueError,KeyError) as error:
        print(f'UPDATE STOPPED: {error}',file=sys.stderr);sys.exit(1)
