import os

for path in [
    r"c:\Users\harsh\Desktop\icps_project\results\final_summary.md",
    r"c:\Users\harsh\Desktop\icps_project\results\heldout_summary.md"
]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update Note
    lines = content.splitlines(keepends=True)
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("> **Delay vs DoS Clarification:"):
            # Replace clarification block
            new_lines.append(
                "> **Telemetry Freeze (`delay` and `dos`) Clarification:** Both `telemetry_freeze (delay)` and `telemetry_freeze (dos)` run the exact same stale-hold code path in `grid_env.py` (freezing sensor measurements to `last_valid_measured_freq` and incrementing `packet_age`). The configured latency (ms) and drop-rate parameters are not used in `step()`, and the held-out delay/dos ranges therefore tested nothing. Both columns are reported side-by-side to maintain historical benchmark alignment.\n"
            )
            i += 1
        elif "| `delay` | `dos` |" in line:
            new_line = line.replace("| `delay` | `dos` |", "| `telemetry_freeze (delay)` | `telemetry_freeze (dos)` |")
            new_lines.append(new_line)
            i += 1
        else:
            new_lines.append(line)
            i += 1

    updated_content = "".join(new_lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(updated_content)
    print(f"Successfully updated {path}")

# Also update final_eval.py build_markdown_summary
eval_script = r"c:\Users\harsh\Desktop\icps_project\final_eval.py"
with open(eval_script, "r", encoding="utf-8") as f:
    eval_text = f.read()

old_eval_note = (
    '> **Delay vs DoS Clarification:** `delay` (latency blackout) and `dos` (telemetry drop) are reported separately for backward benchmark consistency, but share one underlying stale-telemetry mechanism in `grid_env.py` (freezing sensor measurements to `last_valid_measured_freq` and incrementing `packet_age`). Their results should be read as two realizations of the same stale-telemetry phenomenon.'
)
new_eval_note = (
    '> **Telemetry Freeze (`delay` and `dos`) Clarification:** Both `telemetry_freeze (delay)` and `telemetry_freeze (dos)` run the exact same stale-hold code path in `grid_env.py` (freezing sensor measurements to `last_valid_measured_freq` and incrementing `packet_age`). The configured latency (ms) and drop-rate parameters are not used in `step()`, and the held-out delay/dos ranges therefore tested nothing. Both columns are reported side-by-side to maintain historical benchmark alignment.'
)
if old_eval_note in eval_text:
    eval_text = eval_text.replace(old_eval_note, new_eval_note)

old_hdr = 'header = f"| Method | " + " | ".join([f"`{at}`" for at in ATTACK_TYPES]) + " |"'
new_hdr = '''def col_header(at: str) -> str:
            if at == "delay":
                return "`telemetry_freeze (delay)`"
            elif at == "dos":
                return "`telemetry_freeze (dos)`"
            return f"`{at}`"
        header = f"| Method | " + " | ".join([col_header(at) for at in ATTACK_TYPES]) + " |"'''

if old_hdr in eval_text:
    eval_text = eval_text.replace(old_hdr, new_hdr)

old_rl_hdr = 'lines.append("| RL Seed | " + " | ".join([f"`{at}`" for at in ATTACK_TYPES]) + " |\\n")'
new_rl_hdr = 'lines.append("| RL Seed | " + " | ".join([col_header(at) for at in ATTACK_TYPES]) + " |\\n")'
if old_rl_hdr in eval_text:
    eval_text = eval_text.replace(old_rl_hdr, new_rl_hdr)

with open(eval_script, "w", encoding="utf-8") as f:
    f.write(eval_text)
print(f"Successfully updated {eval_script}")
