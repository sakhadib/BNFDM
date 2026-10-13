#!/usr/bin/env bash
# Run ACL pubcheck against the paper.
#
# ACL pubcheck is written for the CAMERA-READY build. Our submission build uses
# \usepackage[review]{acl}, which prints line numbers in the margins, and the
# margin checker reports every one of them as a violation. So this script
# derives a camera-ready-style copy of the source, builds that, and checks it.
# The submission PDF in paper/ is never touched.
#
#   tools/pubcheck.sh            # build the check copy and check it
#   tools/pubcheck.sh --refs     # also run the (CLI-hidden) bibliography check
#
# Exit status is 0 when pubcheck says "All Clear!", 1 otherwise.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT/paper"
SRC="$PAPER_DIR/acl_latex.tex"
WORK="${PUBCHECK_WORKDIR:-$(mktemp -d)}"
VENV="${PUBCHECK_VENV:-$WORK/venv}"
REFS=0
[ "${1:-}" = "--refs" ] && REFS=1

# Print the reason plainly; also emit a GitHub Actions annotation if one day this
# runs in CI, where step logs are awkward to read.
die() {
  echo "pubcheck: $1"
  [ -n "${GITHUB_ACTIONS:-}" ] && echo "::error title=pubcheck::$1"
  exit "${2:-2}"
}

command -v xelatex >/dev/null || die "xelatex not found"
[ -f "$SRC" ] || die "$SRC not found"

# aclpubcheck's deps need setuptools<60 (distutils install_layout), so it gets
# its own venv rather than the ambient interpreter.
if [ ! -x "$VENV/bin/aclpubcheck" ]; then
  echo "pubcheck: installing aclpubcheck into $VENV"
  python3 -m venv "$VENV" || die "could not create the virtualenv at $VENV"
  "$VENV/bin/pip" install -q --upgrade pip || true
  "$VENV/bin/pip" install -q "setuptools<60" wheel \
    || die "could not install setuptools<60 into $VENV"
  # aclpubcheck and its tsv dep use a setup.py that newer setuptools rejects
  # (AttributeError: install_layout), hence the pin above and no build isolation.
  "$VENV/bin/pip" install --no-build-isolation git+https://github.com/acl-org/aclpubcheck \
    || die "could not install aclpubcheck into $VENV"
fi

echo "pubcheck: building a camera-ready copy in $WORK"
mkdir -p "$WORK/build"
sed 's/\\usepackage\[review\]{acl}/\\usepackage{acl}/' "$SRC" > "$WORK/build/check.tex"
for f in figures acl.sty acl_natbib.bst references.bib custom.bib kalpurush.ttf; do
  [ -e "$PAPER_DIR/$f" ] && cp -r "$PAPER_DIR/$f" "$WORK/build/"
done

cd "$WORK/build" || exit 2
xelatex -interaction=nonstopmode check.tex >/dev/null 2>&1
bibtex  check                              >/dev/null 2>&1
xelatex -interaction=nonstopmode check.tex >/dev/null 2>&1
xelatex -interaction=nonstopmode check.tex >"$WORK/build.log" 2>&1
if [ ! -f check.pdf ]; then
  echo "--- LaTeX errors ---"
  grep -A3 '^!' "$WORK/build.log" | head -40
  echo "--- missing files, if any ---"
  grep -iE "not found|cannot find|undefined control sequence|file .* not found" \
       "$WORK/build.log" | sort -u | head -20
  echo "--- tail of the log ---"
  tail -25 "$WORK/build.log"
  first=$(grep -m1 -A2 '^!' "$WORK/build.log" | tr '\n' ' ' | cut -c1-200)
  die "the check copy did not build. First LaTeX error: ${first:-none captured}"
fi

# Guard the things pubcheck cannot see, which matter for the submission build.
echo "pubcheck: submission-build guards"
status=0
grep -q '\\usepackage\[review\]{acl}' "$SRC" \
  && echo "  ok    submission source is still in review mode" \
  || { echo "  FAIL  submission source is not in review mode"; status=1; }
grep -q 'Anonymous' "$SRC" \
  && echo "  ok    author block is anonymous" \
  || { echo "  FAIL  author block is not anonymous"; status=1; }
if grep -q 'Undefined control sequence\|LaTeX Warning: Reference.*undefined\|Citation.*undefined' "$WORK/build.log"; then
  echo "  FAIL  undefined references or citations"; status=1
else
  echo "  ok    no undefined references or citations"
fi

echo "pubcheck: aclpubcheck (long paper)"
if [ "$REFS" = 1 ]; then
  "$VENV/bin/python" - <<'PY' | tee "$WORK/pubcheck.out"
import argparse
from aclpubcheck.formatchecker import Formatter
f = Formatter(argparse.Namespace(num_workers=1, name_check=False, bottom_check=True))
f.format_check("check.pdf", "long", output_dir=".",
               check_references=True, report_name="paper")
PY
else
  "$VENV/bin/aclpubcheck" -p long -o . check.pdf 2>&1 | tee "$WORK/pubcheck.out"
fi

# "All Clear!" means zero errors AND zero warnings. Warnings are advisory, so we
# pass on zero errors and surface the warnings without failing.
if grep -q "All Clear!" "$WORK/pubcheck.out"; then
  nerr=0
else
  nerr=$(sed -n 's/.*We detected \([0-9]\+\) errors.*/\1/p' "$WORK/pubcheck.out" | tail -1)
  [ -z "$nerr" ] && nerr=$(grep -cE '^Error \(' "$WORK/pubcheck.out")
fi

if [ "${nerr:-1}" -eq 0 ]; then
  grep -E '^Warning \(' "$WORK/pubcheck.out" | sed 's/^/  /'
  echo "pubcheck: PASS ($nerr errors)"
else
  grep -E '^(Error|Parsing Error)' "$WORK/pubcheck.out" | sed 's/^/  /'
  echo "pubcheck: FAIL ($nerr errors)"
  status=1
fi
exit $status
