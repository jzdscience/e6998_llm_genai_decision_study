"""Expose original runtime metadata while keeping the new checkout authoritative.

The frozen pip inventory includes the original project's egg-info. Make that
existing metadata discoverable to the unchanged verifier, never filter its output
or replace its expected digest. Only the two auxiliary controllers use this helper.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import threading

from .common import StudyError


@contextmanager
def legacy_environment(binding, project):
    """Scope cwd and import paths for reused environments and new controller code.

    pip may remove the initial cwd from its search path. Explicit PYTHONPATH makes
    original egg-info discovery independent of that behavior. The new checkout is
    both cwd and the first import path, including for Python -m/-c children. This
    main-thread context is for a single CLI controller; all process state is restored
    on exit. No installed metadata, source files or freeze bytes are modified.
    """
    if threading.current_thread() is not threading.main_thread():
        raise StudyError("Legacy runtime context requires the single controller main thread")
    project = Path(project).absolute().resolve()
    source = Path(binding["v1_dir"]).absolute()
    work = Path(binding["source_work_dir"]).absolute()
    if source.parent.name != "results" or work != source/"runtime/work":
        raise StudyError("Cannot identify the original project from the recorded runtime layout")
    original_project = source.parent.parent
    if not source.is_dir() or not original_project.is_dir() or original_project.resolve() == project:
        raise StudyError("Original runtime project and source checkout must exist separately")
    import decision_study
    from . import runtime as frozen
    if (Path(decision_study.__file__).resolve().parent != project/"decision_study"
            or Path(frozen.__file__).resolve() != project/"decision_study/runtime.py"):
        raise StudyError("Controller imports do not belong to the intended source checkout")
    previous_cwd, previous_path = Path.cwd(), list(sys.path)
    previous_pythonpath = os.environ.get("PYTHONPATH")
    # Freeze relative entries before chdir, including the empty interactive entry.
    # The explicit first entry also protects lazy imports of scripts/*.
    pinned_path = [str(project)] + [str((previous_cwd/(entry or ".")).resolve())
                                   if isinstance(entry, str) and not Path(entry).is_absolute()
                                   else entry for entry in previous_path]
    inherited = [] if previous_pythonpath is None else previous_pythonpath.split(os.pathsep)
    child_paths = [str(project), str(original_project)] + [
        str((previous_cwd / (entry or ".")).resolve()) for entry in inherited]
    try:
        sys.path[:] = pinned_path
        os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(child_paths))
        os.chdir(project)
        yield {"original_metadata_root": str(original_project), "controller_source": str(project),
               "subprocess_cwd": str(project)}
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
        if previous_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous_pythonpath
