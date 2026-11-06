"""Read-only local compatibility report; never installs or changes packages."""
from __future__ import annotations
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def windows_hardware():
    if platform.system() != 'Windows':
        return {'note': 'Detailed Windows hardware query applies only when run on the user computer'}
    query = "$cpu=Get-CimInstance Win32_Processor; $system=Get-CimInstance Win32_ComputerSystem; $gpu=Get-CimInstance Win32_VideoController; [pscustomobject]@{cpu_name=@($cpu.Name);physical_cores=($cpu|Measure-Object NumberOfCores -Sum).Sum;logical_cores=($cpu|Measure-Object NumberOfLogicalProcessors -Sum).Sum;ram_bytes=$system.TotalPhysicalMemory;gpu_name=@($gpu.Name)}|ConvertTo-Json -Compress"
    try:
        result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', query], capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            return json.loads(result.stdout)
        return {'hardware_query_error': result.stderr[:500]}
    except Exception as exc:
        return {'hardware_query_error': str(exc)}


def base_code_candidates(project):
    if not project.is_dir():
        return {'project_folder_exists_here': False, 'candidates': []}
    found = []
    for directory, dirs, files in os.walk(project):
        dirs[:] = [x for x in dirs if x not in {'.venv', 'venv', '.git', 'node_modules', 'raw', 'references', '__pycache__'} and not x.startswith('HERA_Exploratory_Training_')]
        for name in files:
            p = Path(directory) / name
            if p.suffix == '.ipynb' or (p.suffix == '.py' and any(s in name.lower() for s in ['train', 'model', 'xgb', 'pipeline'])):
                found.append(str(p.relative_to(project)))
            if len(found) >= 100:
                return {'project_folder_exists_here': True, 'candidates': sorted(found), 'limit_reached': True}
    return {'project_folder_exists_here': True, 'candidates': sorted(found)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=Path(r'V:\Radiation_Profiles_Pfotzer_Peak_Estimation'))
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    versions = {}
    for name in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'xgboost-cpu', 'matplotlib', 'shap', 'jupyterlab', 'ipykernel']:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    result = {'operating_system': platform.platform(), 'architecture': platform.machine(),
              'python_version': platform.python_version(), 'python_executable': sys.executable,
              'logical_cores_python_reports': os.cpu_count(), 'hardware': windows_hardware(),
              'packages': versions, 'recorded_project_folder': str(a.project),
              'base_code_inventory': base_code_candidates(a.project),
              'execution_mode': 'CPU hist; GPU and separate SHAP package not required',
              'python_for_supplied_training_code': 'Python 3.10 or newer; exact environment used here is documented separately',
              'package_note': 'Do not replace the existing environment blindly. XGBoost 2.1.4 was used for the supplied fitted models.'}
    text = json.dumps(result, indent=2)
    print(text)
    if a.output:
        if a.output.exists():
            raise SystemExit('Choose a new output file; existing files are never overwritten')
        a.output.write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
