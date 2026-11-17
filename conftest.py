# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import collections
import contextlib
import os
import re
import sys
import sysconfig
import warnings

import pytest

_implicit_markers = {'default'}
_category_markers = {'solver', 'writer'}
_extended_implicit_markers = _implicit_markers.union(_category_markers)


def pytest_configure(config):
    # If the user specified "--solver" or "--writer", then add that
    # logic to the marker expression
    markexpr = config.option.markexpr
    for cat in _category_markers:
        opt = config.getoption('--' + cat)
        if opt:
            if markexpr:
                markexpr = f"({markexpr}) and "
            markexpr += f"{cat}(id='{opt}')"
    # If the user didn't specify a marker expression, then we will
    # select all "default" tests.
    if not markexpr:
        markexpr = 'default'
    config.option.markexpr = markexpr


def pytest_itemcollected(item):
    """Standardize all Pyomo test markers.

    This callback ensures that all unmarked tests, along with all tests
    that are only marked by category markers (e.g., "solver" or
    "writer"), are also marked with the default (implicit) markers
    (currently just "default").

    About category markers
    ----------------------

    We have historically supported "category markers"::

        @pytest.mark.solver("highs")

    Unfortunately, pytest doesn't allow for building marker
    expressions (e.g., for "-m") based on the marker.args.  We will
    map the positional argument (for pytest.mark.solver and
    pytest.mark.writer) to the keyword argument "id".  This will allow
    querying against specific solver interfaces in marker expressions
    with::

        solver(id='highs')

    We will take this opportunity to also set a keyword argument for
    the solver/writer "vendor" (defined as the id up to the first
    underscore).  This will allow running "all Gurobi tests"
    (including, e.g., lp, direct, and persistent) with::

        -m solver(vendor='gurobi')

    As with all pytest markers, these can be combined into more complex
    "marker expressions" using ``and``, ``or``, ``not``, and ``()``.

    """
    markers = list(item.iter_markers())
    if not markers:
        # No markers; add the implicit (default) markers
        for marker in _implicit_markers:
            item.add_marker(getattr(pytest.mark, marker))
        return

    marker_set = {mark.name for mark in markers}
    # If the item is only marked by extended implicit markers (e.g.,
    # solver and/or writer), then make sure it is also marked by all
    # implicit markers (i.e., "default")
    if marker_set.issubset(_extended_implicit_markers):
        for marker in _implicit_markers - marker_set:
            item.add_marker(getattr(pytest.mark, marker))

    # Map any "category" markers (solver or writer) positional arguments
    # to the id keyword, and ensure the 'vendor' keyword is populated
    for mark in markers:
        if mark.name not in _category_markers:
            continue
        if mark.args:
            (_id,) = mark.args
            mark.kwargs['id'] = _id
        if 'vendor' not in mark.kwargs:
            mark.kwargs['vendor'] = mark.kwargs['id'].split("_")[0]


def pytest_addoption(parser):
    """
    Add parser options as shorthand for running tests marked by specific
    solvers or writers.
    """
    parser.addoption(
        "--solver",
        action="store",
        metavar="SOLVER",
        help="Run tests matching the requested SOLVER.",
    )
    parser.addoption(
        "--writer",
        action="store",
        metavar="WRITER",
        help="Run tests matching the requested WRITER.",
    )
    group = parser.getgroup("pyomo deprecation report")
    group.addoption(
        "--deprecation-report",
        action="store_true",
        default=False,
        help="Record every DeprecationWarning, PendingDeprecationWarning, and "
        "FutureWarning raised during collection and testing (including ones "
        "hidden by -W / PYTHONWARNINGS filters) and print a prioritized "
        "summary at the end of the run.  When running under GitHub Actions, "
        "the report is also appended to $GITHUB_STEP_SUMMARY.",
    )
    group.addoption(
        "--deprecation-report-file",
        action="store",
        metavar="PATH",
        default=None,
        help="Also write the deprecation report (Markdown) to PATH.  "
        "Implies --deprecation-report.",
    )


# ------------------------------------------------------------------------------------
# Deprecation report (--deprecation-report)
# ------------------------------------------------------------------------------------
#
# Our CI runs pytest with "-W ignore::Warning", which keeps the logs clean
# but means Python never records the warnings at all.  When the report is
# requested, we insert "always" filters for the deprecation categories just
# inside pytest's own per-test warning context so that every occurrence
# reaches our showwarning() hook.  The hook records the warning and then
# re-applies the *original* filters (-W, PYTHONWARNINGS, ini
# filterwarnings, marks) to decide whether to pass it on (to pytest's
# recorder / stderr), drop it, or raise it.  Console behavior is therefore
# unchanged, including "-W error::DeprecationWarning".

_DEPRECATION_CATEGORIES = (FutureWarning, DeprecationWarning, PendingDeprecationWarning)
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_STDLIB = os.path.normcase(os.path.abspath(sysconfig.get_paths()['stdlib']))
_SITE_PKG_RE = re.compile(r'[\\/](?:site|dist)-packages[\\/]([^\\/]+)')
_THIS_FILE = os.path.normcase(os.path.abspath(__file__))

# Origin tiers (lower == more likely to be ours to fix)
_ORIGIN_LABELS = {
    0: 'Pyomo source',
    1: 'Pyomo tests/examples/docs',
    2: 'third-party, called from Pyomo source',
    3: 'third-party, called from Pyomo tests',
    4: 'third-party / Python internal',
}


def _get_filters():
    # Python 3.14 context-aware warnings keep filters on the context
    getter = getattr(warnings, '_get_filters', None)
    return list(getter() if getter is not None else warnings.filters)


def _classify_file(filename):
    """Return (kind, label) for a source file.

    kind is one of 'pyomo', 'pyomo-tests', 'third-party', 'python', 'other'.
    """
    path = os.path.normcase(os.path.abspath(filename))
    m = _SITE_PKG_RE.search(path)
    if m:
        pkg = m.group(1).split('.')[0].split('-')[0]
        return 'third-party', pkg
    root = os.path.normcase(_REPO_ROOT)
    if path.startswith(root + os.sep):
        rel = os.path.relpath(path, root).replace(os.sep, '/')
        parts = rel.split('/')
        if parts[0] == 'pyomo' and 'tests' not in parts:
            return 'pyomo', rel
        return 'pyomo-tests', rel
    if path.startswith(_STDLIB + os.sep):
        return 'python', 'stdlib'
    return 'other', filename


# "<version>" optionally preceded by a project name, e.g. "3.16",
# "Python 3.16", "NumPy 2.0", "version 1.4"
_VERSION = (
    r'((?:(?!version\b|a\b|the\b)[A-Za-z][\w.-]*\s+)?(?:version\s+)?v?\d+(?:\.\d+)+)'
)
_DEPRECATED_RE = re.compile(
    r'deprecated\s+(?:(?:since|in|as\s+of|from)\s+)?' + _VERSION, re.I
)
_REMOVAL_RE = re.compile(
    r'(?:remov(?:ed|al)|dropped|deleted)\s+(?:in|from|as\s+of|by)\s+' + _VERSION, re.I
)
_VAGUE_REMOVAL_RE = re.compile(
    r'future\s+(?:version|release)|will\s+(?:be\s+removed|no\s+longer|change|'
    r'raise|stop)|\bremov(?:ed|al)\b',
    re.I,
)
_LIFECYCLE_NOTE = (
    "Sorted by: warning type (Future > Deprecation > Pending), then removal "
    "date, then Pyomo involvement, then count."
)


def _clean_version(v):
    v = re.sub(r'\bversion\s+', '', ' '.join(v.split()), flags=re.I)
    return v.rstrip('.')


def _lifecycle(category, text):
    """Extract (sort_key, deprecated_in, removed_in) from a warning message."""
    cat_rank = next(
        i for i, c in enumerate(_DEPRECATION_CATEGORIES) if issubclass(category, c)
    )
    m = _DEPRECATED_RE.search(text)
    deprecated = _clean_version(m.group(1)) if m else ''
    m = _REMOVAL_RE.search(text)
    if m:
        removal = _clean_version(m.group(1))
        py = re.fullmatch(r'python\s+3\.(\d+)', removal, re.I)
        if py:
            away = int(py.group(1)) - sys.version_info[1]
            if away > 0:
                removal += f" (in {away} release{'s' if away > 1 else ''})"
            return (cat_rank, 0, away), deprecated, removal
        return (cat_rank, 1, 0), deprecated, removal
    if _VAGUE_REMOVAL_RE.search(text):
        return (cat_rank, 2, 0), deprecated, 'future (unspecified)'
    return (cat_rank, 3, 0), deprecated, ''


class _DeprecationRecorder:
    MAX_EXAMPLES = 3
    MAX_SITES = 50

    def __init__(self):
        # key: (category, message text, filename, lineno)
        self.records = {}
        self.where = '<collection>'
        self._module_index = {}

    # -- recording ---------------------------------------------------------------

    def _trigger_site(self):
        """Nearest frame in the Pyomo repo (outside this file) on the stack."""
        root = os.path.normcase(_REPO_ROOT) + os.sep
        f = sys._getframe(2)
        while f is not None:
            fn = os.path.normcase(os.path.abspath(f.f_code.co_filename))
            if fn.startswith(root) and fn != _THIS_FILE and not _SITE_PKG_RE.search(fn):
                return os.path.relpath(fn, root).replace(os.sep, '/'), f.f_lineno
            f = f.f_back
        return None

    def record(self, message, category, filename, lineno):
        text = str(message)
        key = (category, text, filename, lineno)
        rec = self.records.get(key)
        if rec is None:
            rec = self.records[key] = {
                'count': 0,
                'tests': collections.Counter(),
                'sites': collections.Counter(),
            }
        rec['count'] += 1
        rec['tests'][self.where] += 1
        site = self._trigger_site()
        if site is not None and (
            site in rec['sites'] or len(rec['sites']) < self.MAX_SITES
        ):
            rec['sites'][site] += 1

    # -- emulation of the original filters ----------------------------------------

    def _module_name(self, filename):
        if filename not in self._module_index:
            for name, mod in list(sys.modules.items()):
                f = getattr(mod, '__file__', None)
                if f:
                    self._module_index.setdefault(f, name)
        if filename in self._module_index:
            return self._module_index[filename]
        return filename[:-3] if filename.endswith('.py') else filename

    def _original_action(self, filters, message, category, filename, lineno):
        text = str(message)
        module = None
        for action, msg, cat, mod, ln in filters:
            if msg is not None and not msg.match(text):
                continue
            if not issubclass(category, cat):
                continue
            if mod is not None:
                if module is None:
                    module = self._module_name(filename)
                if not mod.match(module):
                    continue
            if ln and ln != lineno:
                continue
            return action
        return warnings.defaultaction

    @contextlib.contextmanager
    def capture(self, where):
        prev_where, self.where = self.where, where
        with warnings.catch_warnings():
            outer_filters = _get_filters()
            forward = warnings.showwarning
            seen = set()
            for cat in _DEPRECATION_CATEGORIES:
                warnings.filterwarnings('always', category=cat)

            def showwarning(message, category, filename, lineno, file=None, line=None):
                if not issubclass(category, _DEPRECATION_CATEGORIES):
                    # Not ours: the real filters were already applied
                    return forward(message, category, filename, lineno, file, line)
                self.record(message, category, filename, lineno)
                action = self._original_action(
                    outer_filters, message, category, filename, lineno
                )
                if action == 'ignore':
                    return
                if action == 'error':
                    if isinstance(message, Warning):
                        raise message
                    raise category(message)
                text = str(message)
                if action == 'once':
                    key = (text, category)
                elif action == 'module':
                    key = (text, category, filename)
                elif action == 'default':
                    key = (text, category, filename, lineno)
                else:  # 'always' (or anything unexpected)
                    key = None
                if key is not None:
                    if key in seen:
                        return
                    seen.add(key)
                return forward(message, category, filename, lineno, file, line)

            warnings.showwarning = showwarning
            try:
                yield
            finally:
                self.where = prev_where

    # -- reporting ----------------------------------------------------------------

    def entries(self):
        out = []
        for (category, text, filename, lineno), rec in self.records.items():
            kind, label = _classify_file(filename)
            site_kinds = {_classify_file(f)[0] for f, _ in rec['sites']}
            if kind == 'pyomo':
                origin = 0
            elif kind == 'pyomo-tests':
                origin = 1
            elif 'pyomo' in site_kinds:
                origin = 2
            elif rec['sites']:
                origin = 3
            else:
                origin = 4
            if kind in ('pyomo', 'pyomo-tests'):
                where = f"{label}:{lineno}"
            elif kind == 'third-party':
                where = f"{label}: {os.path.basename(filename)}:{lineno}"
            elif kind == 'python':
                where = f"Python: {os.path.basename(filename)}:{lineno}"
            else:
                where = f"{filename}:{lineno}"
            sort_key, deprecated, removal = _lifecycle(category, text)
            out.append(
                {
                    'sort': (sort_key, origin, -rec['count'], text),
                    'category': category.__name__,
                    'message': text,
                    'where': where,
                    'origin': _ORIGIN_LABELS[origin],
                    'deprecated': deprecated,
                    'removal': removal,
                    'count': rec['count'],
                    'tests': rec['tests'],
                    # Only report call sites that add information beyond 'where'
                    'sites': collections.Counter(
                        {
                            site: n
                            for site, n in rec['sites'].items()
                            if f"{site[0]}:{site[1]}" != where
                        }
                    ),
                }
            )
        out.sort(key=lambda e: e['sort'])
        return out

    @staticmethod
    def _sites(entry, n):
        return [f"{f}:{l}" for (f, l), _ in entry['sites'].most_common(n)]

    def write_terminal(self, tr, entries):
        total = sum(e['count'] for e in entries)
        tr.write_sep('=', f"{len(entries)} deprecation warnings ({total} occurrences)")
        if not entries:
            tr.write_line("No deprecation warnings were recorded.")
            return
        tr.write_line(_LIFECYCLE_NOTE)
        for i, e in enumerate(entries, 1):
            tr.write_line('')
            tr.write_line(
                f"[{i}] {e['category']}  x{e['count']} in {len(e['tests'])} test(s)",
                bold=True,
            )
            if e['deprecated']:
                tr.write_line(f"    deprecated: {e['deprecated']}")
            if e['removal']:
                tr.write_line(f"    removal:    {e['removal']}")
            tr.write_line(f"    at:         {e['where']}  ({e['origin']})")
            sites = self._sites(e, self.MAX_EXAMPLES)
            if sites:
                more = len(e['sites']) - len(sites)
                tr.write_line(
                    f"    via:        {', '.join(sites)}"
                    + (f" (+{more} more)" if more > 0 else '')
                )
            tests = [t for t, _ in e['tests'].most_common(self.MAX_EXAMPLES)]
            tr.write_line(f"    e.g.:       {', '.join(tests)}")
            msg = ' '.join(e['message'].split())
            if len(msg) > 300:
                msg = msg[:297] + '...'
            tr.write_line(f"    message:    {msg}")

    def markdown(self, entries, title, limit=None):
        def esc(s):
            return ' '.join(str(s).split()).replace('|', '\\|').replace('`', "'")

        total = sum(e['count'] for e in entries)
        lines = [f"## {title}", '']
        if not entries:
            lines.append("No deprecation warnings were recorded. :tada:")
            return '\n'.join(lines) + '\n'
        lines += [
            f"**{len(entries)}** warnings, **{total}** occurrences. " + _LIFECYCLE_NOTE,
            '',
            "| # | Category | Deprecated | Removal | Origin | Count | Tests "
            "| Location | Triggered from | Message |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        shown = entries if limit is None else entries[:limit]
        for i, e in enumerate(shown, 1):
            msg = esc(e['message'])
            if len(msg) > 500:
                msg = msg[:497] + '...'
            sites = '<br>'.join(f"`{s}`" for s in self._sites(e, self.MAX_EXAMPLES))
            lines.append(
                f"| {i} | {e['category']} | {esc(e['deprecated'])} "
                f"| {esc(e['removal'])} | {e['origin']} "
                f"| {e['count']} | {len(e['tests'])} | `{esc(e['where'])}` "
                f"| {sites} | {msg} |"
            )
        if len(shown) < len(entries):
            lines += [
                '',
                f"_{len(entries) - len(shown)} additional warning(s) omitted "
                "to fit the size limit; run with --deprecation-report locally "
                "for the full list._",
            ]
        return '\n'.join(lines) + '\n'


def _deprecation_recorder(config):
    return getattr(config, '_pyomo_deprecation_recorder', None)


@pytest.hookimpl(trylast=True)
def pytest_sessionstart(session):
    config = session.config
    if config.getoption('--deprecation-report') or config.getoption(
        '--deprecation-report-file'
    ):
        config._pyomo_deprecation_recorder = _DeprecationRecorder()


# Note: pytest's own warnings plugin wraps these hooks with tryfirst=True,
# so these wrappers run *inside* pytest's warning-capture context.
@pytest.hookimpl(hookwrapper=True)
def pytest_collection(session):
    rec = _deprecation_recorder(session.config)
    if rec is None:
        yield
        return
    with rec.capture('<collection>'):
        yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    rec = _deprecation_recorder(item.config)
    if rec is None:
        yield
        return
    with rec.capture(item.nodeid):
        yield


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    rec = _deprecation_recorder(config)
    if rec is None:
        return
    entries = rec.entries()
    rec.write_terminal(terminalreporter, entries)

    title = (
        f"Deprecation warnings: Python {sys.version_info[0]}.{sys.version_info[1]}"
        f" on {sys.platform}"
    )
    path = config.getoption('--deprecation-report-file')
    if path:
        with open(path, 'w') as f:
            f.write(rec.markdown(entries, title))
        terminalreporter.write_line(f"Deprecation report written to {path}")
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        # GitHub limits each step summary to 1 MiB; leave some headroom
        limit = None
        text = rec.markdown(entries, title)
        while len(text.encode()) > 900_000:
            limit = len(entries) // 2 if limit is None else limit // 2
            text = rec.markdown(entries, title, limit)
        with open(summary, 'a') as f:
            f.write(text + '\n')
        terminalreporter.write_line("Deprecation report appended to job summary")
