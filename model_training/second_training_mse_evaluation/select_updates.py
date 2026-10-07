"""Extract key updates from the complete E4 validation report."""

from pathlib import Path


SOURCE = Path(__file__).with_name("e4_validation_table.md")
OUTPUT = Path(__file__).with_name("e4_validation_key_updates.md")
UPDATES = (0, 10, 50, 100, 500, 1000, 2000, 5000, 10000)


def main():
    lines = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if line.startswith("| "):
            update = line.split("|", 2)[1].strip()
            if update.isdigit() and int(update) not in UPDATES:
                continue
        lines.append(line)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
