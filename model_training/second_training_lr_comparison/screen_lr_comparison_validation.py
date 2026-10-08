"""Show selected update rows from the learning-rate validation report."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "lr_comparison_validation.md"
OUTPUT = ROOT / "lr_comparison_validation_screened.md"
UPDATES = {0, 10, 50, 100, 1000, 2000, 5000, 10000}
UPDATE_ROW = re.compile(r"^\|\s*(\d+)\s*\|")


def main():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    selected = []
    for line in lines:
        match = UPDATE_ROW.match(line)
        if match is None or int(match.group(1)) in UPDATES:
            selected.append(line)
    OUTPUT.write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
