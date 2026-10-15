#!/usr/bin/env python3
"""Derive paper/arxiv.tex from paper/acl_latex.tex.

acl_latex.tex is the anonymous ARR source and stays the single place where the
body text is edited. This script re-applies the preprint-only changes on top of
it, so the two files can never drift:

  - \\usepackage{acl} instead of [review]: no line numbers, no anonymity banner
  - fontawesome5, for the two link icons in the author block
  - a \\blfootnote macro for the unnumbered equal-contribution note
  - the real author block, laid out as in our LLM-Ideoplasticity preprint
  - the anonymous.4open.science links replaced by the public repository and the
    project page

Run it from anywhere; it resolves paths relative to the repository root.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "paper" / "acl_latex.tex"
DST = ROOT / "paper" / "arxiv.tex"

REPO = "https://github.com/sakhadib/literal-lock-in"
SITE = "https://sakhadib.github.io/literal-lock-in"

AUTHOR_BLOCK = r"""\author{
    Adib Sakhawat\textsuperscript{$*$}
    \quad
    Syed Rifat Raiyan\textsuperscript{$*$}
    \\
    Systems and Software Lab (SSL)
    \\
    Department of Computer Science and Engineering
    \\
    Islamic University of Technology, Dhaka, Bangladesh
    \\
    \texttt{\{adibsakhawat, rifatraiyan\}@iut-dhaka.edu}
    \\[3pt]
    \href{%s}{\faGithub{}~\texttt{sakhadib/literal-lock-in}}
    \quad
    \href{%s}{\faGlobe{}~\texttt{sakhadib.github.io/literal-lock-in}}
}

\begin{document}
\maketitle

\blfootnote{\textsuperscript{$*$}Equal contribution.}
""" % (REPO, SITE)

BLFOOTNOTE = r"""% ── Unnumbered footnote, for the equal-contribution note ─────────────
\newcommand{\blfootnote}[1]{%
    \begingroup
    \renewcommand\thefootnote{}\footnote{#1}%
    \addtocounter{footnote}{-1}%
    \endgroup
}

% ── Title ─────────────────────────────────────────────────────────────"""

EDITS = [
    (r"\usepackage[review]{acl}", r"\usepackage{acl}"),
    (r"\usepackage{booktabs}", "\\usepackage{fontawesome5}\n\\usepackage{booktabs}"),
    (r"% ── Title ─────────────────────────────────────────────────────────────",
     BLFOOTNOTE),
    ("\\author{Anonymous Submission}\n\n\n\n\n\\begin{document}\n\\maketitle\n",
     AUTHOR_BLOCK),
    (r"\url{https://anonymous.4open.science/r/FSS-CC4D}.",
     r"\url{%s}." % REPO),
    ("\\url{https://anonymous.4open.science/r/FSS-CC4D/} for review, and the same material\n"
     "will be released under a permanent identifier on acceptance.",
     "\\url{%s}, with the project page at\n\\url{%s}." % (REPO, SITE)),
]


def main() -> int:
    text = SRC.read_text(encoding="utf-8")
    for old, new in EDITS:
        count = text.count(old)
        if count != 1:
            sys.stderr.write(
                "make_arxiv: expected exactly 1 match, found %d, for:\n  %s\n"
                % (count, old.splitlines()[0][:72])
            )
            return 1
        text = text.replace(old, new)
    for leak in ("Anonymous Submission", "anonymous.4open.science", "[review]"):
        if leak in text:
            sys.stderr.write("make_arxiv: %r survived into the output\n" % leak)
            return 1
    DST.write_text(text, encoding="utf-8")
    print("make_arxiv: wrote %s" % DST.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
