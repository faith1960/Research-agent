#!/usr/bin/env python3
"""
Technical Research Agent
========================
Autonomous agent that researches technical topics and generates structured
markdown reports from web, GitHub, and official manufacturer documentation.

Usage:
  python research_agent.py "Waveshare ESP32-S3-Touch-LCD-1.54 SYS_EN pin"
  python research_agent.py "RP2040 PIO state machine" --output-dir ./reports
  python research_agent.py "STM32F4 USB CDC" --max-turns 60
"""

import anyio
import argparse
import sys
from datetime import datetime
from pathlib import Path
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ResultMessage,
    AssistantMessage,
    TextBlock,
    SystemMessage,
)


# ---------------------------------------------------------------------------
# System prompt — defines research methodology and output structure
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert technical research agent specialized in gathering and
synthesizing hardware and software documentation. You produce rigorous,
citation-backed technical reports.

## Source Priority (highest → lowest)
1. **Official manufacturer datasheets & reference manuals** (direct PDFs, product pages)
2. **Official vendor GitHub repositories** — especially BSP / SDK / driver code,
   open issues, pull requests, and README files
3. **Third-party GitHub repositories** — community drivers, forks, examples
4. **Reputable technical forums** — Stack Overflow, ESP32 forums, Arduino forums,
   Reddit r/esp32, EEVblog, Hackaday
5. **Tutorials & blogs** — only when they cite primary sources or provide unique
   technical detail not found elsewhere

## Research Process
For each topic you receive:

### Step 1 — Orientation (2-3 searches)
- Identify the exact product / IC / library involved
- Find the manufacturer and their official documentation URL
- Find the primary GitHub repository

### Step 2 — Deep Dive (parallel tracks)
**Track A — Official docs:**
  - Fetch datasheet / reference manual
  - Note pin names, voltages, logic levels, protocols, timing

**Track B — GitHub:**
  - Search for BSP / board support package code
  - Read header files for pin definitions and register maps
  - Scan open & closed issues for known bugs, workarounds, undocumented behavior
  - Check PRs for patches that have not yet been merged

**Track C — Community:**
  - Search forums for practical usage notes and confirmed workarounds
  - Note post dates; prefer recent (< 2 years) sources

### Step 3 — Link Following (max 3 levels deep)
When a page references another relevant technical resource, follow that link.
Keep a mental stack of depth: starting URL = level 1, link from there = level 2,
link from that = level 3. Do not go deeper.

### Step 4 — Synthesis
- Cross-reference all sources
- Flag any contradictions explicitly with ❌ CONTRADICTION
- Mark uncertain or unverified claims with ⚠️ UNVERIFIED
- Assign confidence: HIGH (official doc), MEDIUM (2+ independent sources), LOW (single community post)

## Content Filtering Rules
**INCLUDE:**
- Pin numbers, names, voltages, current limits
- Communication protocols (SPI, I2C, UART, etc.) and their parameters
- Register addresses and bit fields
- Initialization sequences and power-on requirements
- Driver / library function signatures and usage examples
- Known silicon errata and workarounds
- Verified working code snippets (with source URL)

**EXCLUDE:**
- Marketing copy, product descriptions, pricing, availability
- Generic "getting started" tutorials that add no technical depth
- Duplicate information already captured from a higher-priority source

## Output Format
Save the report as a markdown file. The structure MUST be:

```
# Technical Research Report: <topic>
**Generated:** <ISO timestamp>
**Research Depth:** up to 3 link levels

---

## 1. Executive Summary
<!-- 3-5 sentence technical summary -->

## 2. Sources
| # | URL | Type | Reliability | Notes |
|---|-----|------|-------------|-------|

## 3. Technical Specifications
<!-- Pins, voltages, protocols, timing — use tables where possible -->

## 4. Pin Reference
<!-- If hardware: complete pinout table with function, direction, voltage -->

## 5. Code Examples
<!-- Verified examples with source attribution -->

## 6. Known Issues & Workarounds
<!-- Bugs, errata, undocumented behavior, community-confirmed workarounds -->

## 7. Contradictions & Discrepancies
<!-- ❌ CONTRADICTION entries, or "None found." -->

## 8. Open Questions
<!-- Items that could not be resolved with available sources -->

## 9. Recommendations
<!-- Practical advice for someone using this hardware/software -->
```

Always write the complete report to the file path given to you.
Do NOT truncate — write every section even if some are brief.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_filename(topic: str) -> str:
    """Convert a free-form topic string to a safe filename fragment."""
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in topic)
    return safe.replace(" ", "_")[:80].strip("_")


def _truncate(text: str, max_chars: int = 120) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "…"


# ---------------------------------------------------------------------------
# Core research routine
# ---------------------------------------------------------------------------

async def research_topic(topic: str, output_dir: Path, max_turns: int) -> Path:
    """
    Launch the research agent for *topic*, stream progress to stdout,
    and return the path of the saved report.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"research_{sanitize_filename(topic)}_{timestamp}.md"
    output_path = output_dir / filename

    # ── Banner ────────────────────────────────────────────────────────────
    sep = "=" * 64
    print(f"\n{sep}")
    print("  TECHNICAL RESEARCH AGENT")
    print(sep)
    print(f"  Topic      : {topic}")
    print(f"  Report     : {output_path}")
    print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Max turns  : {max_turns}")
    print(f"{sep}\n")

    # ── Prompt ────────────────────────────────────────────────────────────
    prompt = f"""\
Research the following technical topic and produce a complete technical report.

**Topic**: {topic}

## Instructions

1. **Orientation** — run 2-3 web searches to identify the manufacturer,
   official datasheet URL, and primary GitHub repository.

2. **Official docs** — fetch the datasheet / reference manual / product page.
   Extract all pin assignments, electrical specs, and protocol details.

3. **GitHub research** — search for:
   - BSP / board support package repositories
   - Driver libraries and their header files
   - Open and closed issues mentioning this component
   - Any PRs with relevant patches

4. **Community research** — search forums and Stack Overflow for practical
   usage notes, confirmed bugs, and workarounds.

5. **Follow links** — if a page references another relevant technical doc,
   fetch it (max 3 levels deep from each starting URL).

6. **Synthesise** — cross-reference all sources. Flag contradictions with
   ❌ CONTRADICTION and unverified claims with ⚠️ UNVERIFIED.

7. **Write report** — save the complete markdown report to exactly this path:
   {output_path}

Priority order: official docs > GitHub BSP > GitHub community > forums.
Do not truncate any section. Write all 9 sections even if brief.
"""

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=["WebSearch", "WebFetch", "Write", "Read"],
        permission_mode="acceptEdits",
        max_turns=max_turns,
        cwd=str(output_dir),
    )

    # ── Stream agent output ───────────────────────────────────────────────
    print("Research in progress…\n")
    turn = 0

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, SystemMessage):
            if message.subtype == "init":
                sid = message.data.get("session_id", "?")
                print(f"  [session {sid[:8]}…]\n")

        elif isinstance(message, AssistantMessage):
            turn += 1
            for block in message.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    first_line = block.text.strip().splitlines()[0]
                    print(f"  [{turn:02d}] {_truncate(first_line)}")

        elif isinstance(message, ResultMessage):
            print(f"\n{sep}")
            print("  RESEARCH COMPLETE")
            print(f"{sep}")

    # ── Verify output ─────────────────────────────────────────────────────
    if output_path.exists():
        size = output_path.stat().st_size
        lines = output_path.read_text(encoding="utf-8").splitlines()
        print(f"\n✓  Report saved  : {output_path}")
        print(f"   Size          : {size:,} bytes")
        print(f"   Lines         : {len(lines)}")

        print(f"\n{'─'*64}")
        print("  REPORT PREVIEW (first 30 lines)")
        print(f"{'─'*64}")
        for line in lines[:30]:
            print(line)
        if len(lines) > 30:
            print(f"\n  … {len(lines) - 30} more lines …")
        print(f"\n  Full report → {output_path}")
    else:
        print(f"\n⚠  Report file not found at: {output_path}")
        print("   The agent may have saved it under a different name.")
        # List any .md files created in the session
        md_files = sorted(output_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if md_files:
            print(f"\n   Most recent .md files in {output_dir}:")
            for f in md_files[:5]:
                print(f"     {f}")

    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Technical Research Agent — autonomous documentation gatherer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python research_agent.py "Waveshare ESP32-S3-Touch-LCD-1.54 SYS_EN pin"
  python research_agent.py "RP2040 PIO state machine programming"
  python research_agent.py "STM32F4 USB CDC implementation" --output-dir ./reports
  python research_agent.py "CH341 USB-Serial driver Linux" --max-turns 60
""",
    )
    parser.add_argument(
        "topic",
        help='Technical topic to research (quote multi-word topics)',
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./reports",
        metavar="DIR",
        help="Directory for saved reports (default: ./reports)",
    )
    parser.add_argument(
        "--max-turns", "-t",
        type=int,
        default=50,
        metavar="N",
        help="Maximum agent turns / tool calls (default: 50)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        anyio.run(research_topic, args.topic, output_dir, args.max_turns)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
