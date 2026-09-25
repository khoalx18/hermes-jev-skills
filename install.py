#!/usr/bin/env python3
"""Install Hermes Jev Skills for whichever agents live on this machine.

    python3 install.py                 # detect Hermes / Claude Code / Codex and install for each
    python3 install.py --check         # show what would happen, change nothing
    python3 install.py --uninstall

On Hermes it installs every plugin under hermes/plugin, the skills, and the scripts in
hermes/scripts; --uninstall removes those and nothing else.

It never asks for, reads or prints an API key. Connecting the key is a separate,
private step: `jev setup-key`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

REPO = Path(__file__).resolve().parent
PLUGIN_SOURCE = REPO / "hermes" / "plugin"
SCRIPT_SOURCE = REPO / "hermes" / "scripts"
# Discovered, never listed. The installer used to name one plugin, so hermes-handoff shipped
# for weeks in a state where nobody who followed the docs could install it.
PLUGINS = sorted(p.name for p in PLUGIN_SOURCE.iterdir() if (p / "plugin.yaml").is_file())
SCRIPTS = sorted(p.name for p in SCRIPT_SOURCE.iterdir() if p.is_file() and not p.name.startswith("."))
SKILLS = sorted(p.name for p in (REPO / "skills").iterdir() if (p / "SKILL.md").is_file())


def _copytree(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))


def _copyfile(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.is_dir():
        _remove(dst)
    shutil.copy2(src, dst)  # copy2 keeps the executable bit: these scripts are run from cron or launchd


def _link(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        shutil.rmtree(link)
    try:
        link.symlink_to(target)
    except OSError:
        # Windows without Developer Mode / privilege: fall back to copy
        if target.is_dir():
            shutil.copytree(target, link, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
        else:
            shutil.copy2(target, link)


def _remove(path: Path) -> bool:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return True
    if path.is_dir():
        shutil.rmtree(path)
        return True
    return False


# ── Hermes ───────────────────────────────────────────────────────────────────

def hermes_homes(root: Path) -> List[Path]:
    homes = [root]
    profiles = root / "profiles"
    if profiles.is_dir():
        homes += sorted(p for p in profiles.iterdir() if (p / "config.yaml").is_file())
    return homes


def enable_plugins(config: Path, names: Sequence[str], enable: bool) -> Dict[str, str]:
    """Add or remove each plugin under plugins.enabled by editing only that list.

    A text edit, not a YAML round-trip: comments, ordering and every other setting survive.
    One pass for all of them, so a two-plugin install leaves one backup rather than two.
    """
    text = config.read_text(encoding="utf-8")
    lines = text.split("\n")
    status: Dict[str, str] = {}
    try:
        start = next(i for i, line in enumerate(lines) if line.rstrip() == "plugins:")
    except StopIteration:
        if not enable:
            return {name: "no plugins section" for name in names}
        if lines and lines[-1] == "":
            lines.pop()                        # the file's own final newline, put back below
        lines += ["plugins:", "  enabled:"] + [f"  - {name}" for name in sorted(names)] + [""]
        status = {name: "enabled" for name in names}
        start = None
    if start is not None:
        end = next((i for i in range(start + 1, len(lines)) if lines[i] and not lines[i].startswith((" ", "#"))), len(lines))
        block = lines[start + 1:end]
        key = next((i for i, line in enumerate(block) if re.match(r"^  enabled:\s*(\[\s*\])?\s*$", line)), None)
        # Every plugin goes in at the same spot, so walking the names backwards leaves
        # them alphabetical in the file.
        for name in sorted(names, reverse=True):
            item = re.compile(rf"^\s*-\s*['\"]?{re.escape(name)}['\"]?\s*$")
            present = [i for i, line in enumerate(block) if item.match(line)]
            if enable:
                if present:
                    status[name] = "already enabled"
                    continue
                if key is None:
                    block.insert(0, "  enabled:")
                    key = 0
                block[key] = "  enabled:"          # turns `enabled: []` into a block list
                block.insert(key + 1, f"  - {name}")
                status[name] = "enabled"
            else:
                if not present:
                    status[name] = "was not enabled"
                    continue
                for i in reversed(present):
                    del block[i]
                status[name] = "disabled"
        lines[start + 1:end] = block
        status = {name: status[name] for name in sorted(status)}  # report reads like the file
    if not any(state in ("enabled", "disabled") for state in status.values()):
        return status                          # nothing moved, so no backup and no rewrite
    backup = config.with_name(f"{config.name}.bak-jev-{time.strftime('%Y%m%dT%H%M%S')}")
    shutil.copy2(config, backup)
    temp = config.with_name(config.name + ".jev-tmp")
    temp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temp, config)
    return status


def install_hermes(root: Path, enable: str, check: bool) -> Dict[str, object]:
    homes = hermes_homes(root)
    if enable == "all":
        wanted = {"default"} | {h.name for h in homes[1:]}
    elif enable == "none":
        wanted = set()
    else:
        wanted = set(filter(None, enable.split(",")))
    report: Dict[str, object] = {
        "home": str(root),
        "profiles": len(homes) - 1,
        "plugins": {name: str(root / "plugins" / name) for name in PLUGINS},
        "scripts": [str(root / "scripts" / name) for name in SCRIPTS],
        "enabled_in": {},
    }
    if check:
        report["would_enable_in"] = sorted(wanted)
        return report
    for name in PLUGINS:
        plugin_dir = root / "plugins" / name
        _copytree(PLUGIN_SOURCE / name, plugin_dir)
        # A copy of jevkit per plugin, because a plugin can be loaded alone: nightly-handoff.py
        # puts only the hermes-handoff directory on sys.path and imports jevkit from there.
        _copytree(REPO / "jevkit", plugin_dir / "jevkit")
    skills_dir = root / "skills" / "jev"
    skills_dir.mkdir(parents=True, exist_ok=True)
    for name in SKILLS:
        _copytree(REPO / "skills" / name, skills_dir / name)
    for name in SCRIPTS:
        _copyfile(SCRIPT_SOURCE / name, root / "scripts" / name)
    for home in homes[1:]:
        for name in PLUGINS:
            _link(root / "plugins" / name, home / "plugins" / name)   # every lane scans its OWN plugins folder
        _link(skills_dir, home / "skills" / "jev")
    for home in homes:
        label = "default" if home == root else home.name
        if label in wanted and (home / "config.yaml").is_file():
            report["enabled_in"][label] = enable_plugins(home / "config.yaml", PLUGINS, True)  # type: ignore[index]
    return report


def uninstall_hermes(root: Path) -> Dict[str, object]:
    removed = []
    for home in hermes_homes(root):
        if (home / "config.yaml").is_file():
            enable_plugins(home / "config.yaml", PLUGINS, False)
        for path in [home / "plugins" / name for name in PLUGINS] + [home / "skills" / "jev"]:
            if _remove(path):
                removed.append(str(path))
    for name in SCRIPTS:                       # installed at the root only, so removed there only
        if _remove(root / "scripts" / name):
            removed.append(str(root / "scripts" / name))
    try:
        (root / "scripts").rmdir()             # goes only if empty: a Hermes home often keeps its own scripts here
    except OSError:
        pass
    return {"removed": removed}


# ── skill folders (Claude Code, Codex, generic) ──────────────────────────────

def install_skills(folder: Path, check: bool) -> Dict[str, object]:
    if not check:
        folder.mkdir(parents=True, exist_ok=True)
        for name in SKILLS:
            _copytree(REPO / "skills" / name, folder / name)
    return {"folder": str(folder), "skills": SKILLS}


def install_cli(check: bool) -> Dict[str, object]:
    target = Path.home() / ".local" / "bin" / "jev"
    if not check:
        _link(REPO / "bin" / "jev", target)
    on_path = str(target.parent) in os.environ.get("PATH", "").split(os.pathsep)
    return {"command": str(target), "on_path": on_path}


def path_warning(cli: Dict[str, object]) -> str | None:
    """Warn when ~/.local/bin is not on PATH, which it is not by default on macOS or most Linux.

    This lived inside the ``cli`` object, where nobody read it, and the very next command
    the docs give a person — ``jev setup-key`` — died with "command not found".
    """
    if cli["on_path"]:
        return None
    command = Path(str(cli["command"]))
    return (f"The `jev` command goes to {command}, but {command.parent} is not on PATH, so "
            f"`jev setup-key` and every other `jev` command will fail with command not found. "
            f"Either add {command.parent} to PATH in your shell profile, or run "
            f"{REPO / 'bin' / 'jev'} everywhere the docs say `jev`.")


def nothing_installed_warning(hermes: Path, check: bool) -> str:
    """Say plainly that a machine with no agent on it got nothing but the CLI.

    Without this the run looked like every successful one: exit 0, success-shaped JSON,
    and a next-steps list for a Hermes that is not there.
    """
    tense = "would be installed" if check else "was installed"
    return (f"No agent was found on this machine: no Hermes home at {hermes}, and no Claude "
            f"Code, Codex or generic skills folder, so nothing {tense} except the `jev` command "
            f"itself. If your agent reads skills from somewhere else, install them there with "
            f"--skills-dir <path>.")


def home_warning(hermes: Path) -> str | None:
    """Warn when the resolved Hermes home is a single profile, not the fleet root.

    Agents run with ``HERMES_HOME`` set to their own profile directory, so a bare
    ``python3 install.py`` from an agent shell installs for that lane only and every
    other lane keeps the old copy.
    """
    if any(part == "profiles" for part in hermes.parts):
        return (f"HERMES_HOME resolved to a profile home ({hermes}), so this install "
                f"covers that lane only. For the whole fleet pass "
                f"--hermes-home {Path.home() / '.hermes'}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--hermes-home", default=None,
                        help="Hermes root (default: $HERMES_HOME, else ~/.hermes)")
    parser.add_argument("--enable", default="all", help="Hermes profiles to enable the plugins in: all, none, or a,b,c")
    parser.add_argument("--skills-dir", action="append", default=[], help="extra skill folder to install into")
    args = parser.parse_args()

    home = Path.home()
    hermes = Path(args.hermes_home).expanduser() if args.hermes_home else Path(
        os.environ.get("HERMES_HOME") or str(home / ".hermes")).expanduser()
    folders = [Path(p).expanduser() for p in args.skills_dir]
    folders += [p for p in (home / ".claude" / "skills", home / ".codex" / "skills", home / ".agents" / "skills") if p.parent.is_dir()]

    report: Dict[str, object] = {"repo": str(REPO), "mode": "uninstall" if args.uninstall else "check" if args.check else "install"}
    warnings: List[str] = [w for w in (home_warning(hermes),) if w]
    if args.uninstall:
        if hermes.is_dir():
            report["hermes"] = uninstall_hermes(hermes)
        report["skills_removed"] = [str(f / n) for f in folders for n in SKILLS if _remove(f / n)]
        _remove(home / ".local" / "bin" / "jev")
    else:
        cli = install_cli(args.check)
        report["cli"] = cli
        warnings += [w for w in (path_warning(cli),) if w]
        if (hermes / "config.yaml").is_file():
            report["hermes"] = install_hermes(hermes, args.enable, args.check)
        report["skill_folders"] = [install_skills(f, args.check) for f in folders]
        steps = ["jev doctor", "jev setup-key   (only if the key is missing; the person pastes it in a private page)",
                 "jev models suggest --write   (only if no routing pools exist yet)"]
        if "hermes" in report:
            steps.append("Hermes: restart the gateway when convenient, then /jev routing shadow")
        elif not folders:
            warnings.append(nothing_installed_warning(hermes, args.check))
        report["next"] = steps
    # Warning first, so it is read before the wall of paths underneath it.
    print(json.dumps({**({"warning": "\n".join(warnings)} if warnings else {}), **report}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
