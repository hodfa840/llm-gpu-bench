#!/usr/bin/env python3
"""
Email the GPU benchmark results to a specified address.
Reads all results_*.json files and sends a formatted HTML + plain-text report.

Usage:
  python 05_email_results.py                             # uses default recipient
  python 05_email_results.py --to someone@example.com
  python 05_email_results.py --smtp-host smtp.gmail.com --smtp-port 587 --user u --password p

If no SMTP credentials are given, the script tries the local sendmail/postfix first.
"""

import json
import smtplib
import argparse
import subprocess
import socket
import bench_config          # reads config.env
from pathlib import Path
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.application import MIMEApplication

DEFAULT_TO   = bench_config.get("EMAIL_TO",   "hodfa71@liu.se")
DEFAULT_FROM = bench_config.get("EMAIL_FROM",  f"gpu-bench@{socket.gethostname()}")

# ─── Load results ─────────────────────────────────────────────────────────────

def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def collect_results(result_dir="."):
    files = sorted(Path(result_dir).glob("results_*.json"))
    data  = {}
    for f in files:
        key  = f.stem.replace("results_", "")
        blob = load_json(f)
        if blob:
            data[key] = blob
    return data


# ─── Format plain-text body ───────────────────────────────────────────────────

def fmt_plain(data, hostname):
    lines = []
    lines.append("=" * 70)
    lines.append("  GPU LLM BENCHMARK REPORT")
    lines.append(f"  Host    : {hostname}")
    lines.append(f"  Date    : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    # GPU info from any result
    for key, blob in data.items():
        if "gpu" in blob:
            lines.append(f"\n  GPU     : {blob.get('gpu', '?')}")
            lines.append(f"  VRAM    : {blob.get('vram_gb', '?')} GB")
            if "ram_total_gb" in blob:
                lines.append(f"  RAM     : {blob.get('ram_total_gb', '?')} GB")
            if "cpu_count" in blob:
                lines.append(f"  CPUs    : {blob.get('cpu_count', '?')} logical")
            break

    # ── Raw compute ────────────────────────────────────────────────────────────
    if "raw_benchmark" in data:
        b = data["raw_benchmark"].get("benchmarks", {})
        lines.append("\n" + "─" * 70)
        lines.append("  RAW COMPUTE")
        lines.append("─" * 70)
        for k, v in b.items():
            if isinstance(v, dict):
                val = v.get("tflops") or v.get("tops") or "?"
                lines.append(f"  {k:<12}: {val} TFLOPS/TOPS")
            else:
                lines.append(f"  {k:<12}: {v}")

    # ── LLM inference ──────────────────────────────────────────────────────────
    if "llm_benchmark" in data:
        res = data["llm_benchmark"].get("results", {})
        lines.append("\n" + "─" * 70)
        lines.append("  LLM INFERENCE  (decode tok/s  |  VRAM model  |  RAM used)")
        lines.append("─" * 70)
        lines.append(f"  {'Model':<22}  {'bf16':>10}  {'bf16 VRAM':>10}  {'4-bit':>10}  {'4bit VRAM':>10}  {'RAM':>8}")
        lines.append("  " + "-" * 68)
        for name, mdata in res.items():
            bf  = mdata.get("bf16", {})
            nf4 = mdata.get("4bit_nf4", {})
            bf_tok  = bf.get("decode_tok_s",  "OOM")
            nf4_tok = nf4.get("decode_tok_s", "OOM")
            bf_vram = bf.get("vram_model_gb",  "—")
            nf4_vram= nf4.get("vram_model_gb", "—")
            ram     = nf4.get("ram_used_gb") or bf.get("ram_used_gb") or "—"
            bf_s    = f"{bf_tok:.1f}"   if isinstance(bf_tok,  float) else str(bf_tok)
            nf4_s   = f"{nf4_tok:.1f}"  if isinstance(nf4_tok, float) else str(nf4_tok)
            bf_v    = f"{bf_vram:.1f}G" if isinstance(bf_vram,  float) else str(bf_vram)
            nf4_v   = f"{nf4_vram:.1f}G"if isinstance(nf4_vram,float) else str(nf4_vram)
            ram_s   = f"{ram:.1f}G"     if isinstance(ram,     float) else str(ram)
            lines.append(f"  {name:<22}  {bf_s:>10}  {bf_v:>10}  {nf4_s:>10}  {nf4_v:>10}  {ram_s:>8}")

    # ── Training ───────────────────────────────────────────────────────────────
    if "training_benchmark" in data:
        res = data["training_benchmark"].get("results", [])
        lines.append("\n" + "─" * 70)
        lines.append("  TRAINING THROUGHPUT (fwd + bwd + optimizer)")
        lines.append("─" * 70)
        lines.append(f"  {'Model':<10}  {'Batch':>6}  {'Seq':>5}  {'Tok/s':>10}  {'ms/step':>8}  {'VRAM':>8}")
        lines.append("  " + "-" * 50)
        for r in res:
            lines.append(
                f"  {r.get('config','?'):<10}  "
                f"{r.get('batch_size',0):>6}  "
                f"{r.get('seq_len',0):>5}  "
                f"{r.get('tokens_per_s',0):>10,}  "
                f"{r.get('ms_per_step',0):>8.1f}  "
                f"{r.get('vram_gb',0):>7.2f}G"
            )

    lines.append("\n" + "=" * 70)
    lines.append("  Generated by GPU LLM Benchmark Suite")
    lines.append("=" * 70)
    return "\n".join(lines)


# ─── Format HTML body ─────────────────────────────────────────────────────────

def tok_color(v):
    if not isinstance(v, (int, float)): return "#e67e73"
    if v > 60:  return "#57c965"
    if v > 20:  return "#f0a742"
    return "#e67e73"

def fmt_html(data, hostname):
    gpu = "?"
    for _, blob in data.items():
        if "gpu" in blob:
            gpu = blob["gpu"]
            vram = blob.get("vram_gb", "?")
            ram  = blob.get("ram_total_gb", "?")
            cpus = blob.get("cpu_count", "?")
            break

    rows_inf = ""
    if "llm_benchmark" in data:
        for name, mdata in data["llm_benchmark"].get("results", {}).items():
            bf  = mdata.get("bf16", {})
            nf4 = mdata.get("4bit_nf4", {})
            bf_tok  = bf.get("decode_tok_s",  None)
            nf4_tok = nf4.get("decode_tok_s", None)
            bf_vram  = bf.get("vram_model_gb", None)
            nf4_vram = nf4.get("vram_model_gb", None)
            ram_v    = nf4.get("ram_used_gb") or bf.get("ram_used_gb")
            bf_ttft  = bf.get("ttft_ms", "—")
            nf4_ttft = nf4.get("ttft_ms", "—")

            bf_s  = f"{bf_tok:.1f}"   if isinstance(bf_tok, float)  else "OOM"
            nf4_s = f"{nf4_tok:.1f}"  if isinstance(nf4_tok, float) else "OOM"
            bf_v  = f"{bf_vram:.1f} GB"  if isinstance(bf_vram, float)  else "—"
            nf4_v = f"{nf4_vram:.1f} GB" if isinstance(nf4_vram, float) else "—"
            ram_s = f"{ram_v:.1f} GB" if isinstance(ram_v, float) else "—"
            rows_inf += f"""
            <tr>
              <td><b>{name}</b></td>
              <td style="color:{tok_color(bf_tok)};font-weight:bold">{bf_s}</td>
              <td>{bf_v}</td>
              <td>{bf_ttft} ms</td>
              <td style="color:{tok_color(nf4_tok)};font-weight:bold">{nf4_s}</td>
              <td>{nf4_v}</td>
              <td>{nf4_ttft} ms</td>
              <td>{ram_s}</td>
            </tr>"""

    rows_train = ""
    if "training_benchmark" in data:
        for r in data["training_benchmark"].get("results", []):
            rows_train += f"""
            <tr>
              <td>{r.get('config','?')}</td>
              <td>{r.get('n_params_M','?')} M</td>
              <td>{r.get('batch_size','?')}</td>
              <td>{r.get('seq_len','?')}</td>
              <td style="color:#57c965;font-weight:bold">{r.get('tokens_per_s',0):,}</td>
              <td>{r.get('ms_per_step','?'):.1f}</td>
              <td>{r.get('vram_gb','?'):.2f} GB</td>
            </tr>"""

    raw_rows = ""
    if "raw_benchmark" in data:
        for k, v in data["raw_benchmark"].get("benchmarks", {}).items():
            if isinstance(v, dict):
                val = v.get("tflops") or v.get("tops") or "?"
                unit = "TFLOPS" if "tflops" in v else "TOPS"
            else:
                val, unit = v, ""
            raw_rows += f"<tr><td>{k}</td><td style='color:#57c965;font-weight:bold'>{val}</td><td>{unit}</td></tr>"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body      {{ font-family: 'Segoe UI', Arial, sans-serif; background:#1a1a2e; color:#e0e0e0; margin:0; padding:20px; }}
  .card     {{ background:#16213e; border-radius:12px; padding:24px; margin-bottom:20px; box-shadow:0 4px 20px rgba(0,0,0,.5); }}
  h1        {{ color:#7b8cde; margin-top:0; font-size:1.6em; }}
  h2        {{ color:#a0a8f0; font-size:1.1em; border-bottom:1px solid #2a2a5a; padding-bottom:8px; }}
  table     {{ width:100%; border-collapse:collapse; font-size:.88em; }}
  th        {{ background:#0f3460; color:#c0c8ff; text-align:left; padding:8px 10px; }}
  td        {{ padding:7px 10px; border-bottom:1px solid #1f1f4f; }}
  tr:hover  {{ background:#1e2a55; }}
  .badge    {{ display:inline-block; background:#0f3460; border-radius:6px; padding:4px 10px; margin:4px; font-size:.82em; }}
  .highlight{{ color:#7fffb0; font-weight:bold; }}
  .footer   {{ color:#555; font-size:.78em; text-align:center; margin-top:20px; }}
</style>
</head>
<body>
<div class="card">
  <h1>🖥️ GPU LLM Benchmark Report</h1>
  <span class="badge">🖥️ {gpu}</span>
  <span class="badge">💾 VRAM: {vram} GB</span>
  <span class="badge">🗄️ RAM: {ram} GB</span>
  <span class="badge">🔲 CPUs: {cpus}</span>
  <span class="badge">🏠 Host: {hostname}</span>
  <span class="badge">📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
</div>

{"<div class='card'><h2>⚡ Raw Compute</h2><table><tr><th>Operation</th><th>Result</th><th>Unit</th></tr>" + raw_rows + "</table></div>" if raw_rows else ""}

{"<div class='card'><h2>🚀 LLM Inference — Decode Throughput (tok/s)</h2><table><tr><th>Model</th><th>bf16 tok/s</th><th>bf16 VRAM</th><th>bf16 TTFT</th><th>4-bit tok/s</th><th>4-bit VRAM</th><th>4-bit TTFT</th><th>RAM Used</th></tr>" + rows_inf + "</table><p style='color:#888;font-size:.82em'>★ Higher tok/s = faster generation. TTFT = time-to-first-token.</p></div>" if rows_inf else ""}

{"<div class='card'><h2>🏋️ Training Throughput (fwd + bwd + optimizer)</h2><table><tr><th>Model</th><th>Params</th><th>Batch</th><th>Seq</th><th>Tok/s</th><th>ms/step</th><th>VRAM</th></tr>" + rows_train + "</table></div>" if rows_train else ""}

<div class="footer">Generated by GPU LLM Benchmark Suite — {datetime.now().isoformat()}</div>
</body>
</html>"""
    return html


# ─── Send ─────────────────────────────────────────────────────────────────────

def send_via_smtp(msg, host, port, user, password, use_tls):
    with smtplib.SMTP(host, port) as s:
        s.ehlo()
        if use_tls:
            s.starttls()
            s.ehlo()
        if user and password:
            s.login(user, password)
        s.send_message(msg)


def send_via_sendmail(msg):
    proc = subprocess.Popen(
        ["/usr/sbin/sendmail", "-t", "-oi"],
        stdin=subprocess.PIPE,
    )
    proc.communicate(msg.as_bytes())
    if proc.returncode not in (None, 0):
        raise RuntimeError(f"sendmail exited with {proc.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Email GPU benchmark results")
    parser.add_argument("--to",        default=DEFAULT_TO)
    parser.add_argument("--from-addr", default=DEFAULT_FROM)
    parser.add_argument("--subject",   default=None)
    parser.add_argument("--smtp-host", default=bench_config.get("SMTP_HOST") or None)
    parser.add_argument("--smtp-port", type=int, default=bench_config.get_int("SMTP_PORT", 587))
    parser.add_argument("--user",      default=bench_config.get("SMTP_USER")   or None)
    parser.add_argument("--password",  default=bench_config.get("SMTP_PASSWORD") or None)
    parser.add_argument("--no-tls",    action="store_true",
                        default=bench_config.get_bool("SMTP_NO_TLS", False))
    parser.add_argument("--result-dir",default=".")
    args = parser.parse_args()

    hostname = socket.gethostname()
    subject  = args.subject or f"GPU Benchmark Results — {hostname} — {torch_gpu_name()}"

    print(f"  Collecting results from '{args.result_dir}'…")
    data = collect_results(args.result_dir)
    if not data:
        print("  ⚠  No results_*.json files found. Run the benchmarks first.")
        return

    print(f"  Found: {', '.join(data.keys())}")

    plain = fmt_plain(data, hostname)
    html  = fmt_html(data, hostname)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = args.from_addr
    msg["To"]      = args.to
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html,  "html"))

    # Attach JSON files
    for f in sorted(Path(args.result_dir).glob("results_*.json")):
        with open(f, "rb") as fh:
            part = MIMEApplication(fh.read(), Name=f.name)
            part["Content-Disposition"] = f'attachment; filename="{f.name}"'
            msg.attach(part)

    print(f"  Sending to {args.to}…")
    try:
        if args.smtp_host:
            send_via_smtp(msg, args.smtp_host, args.smtp_port,
                          args.user, args.password, not args.no_tls)
        else:
            send_via_sendmail(msg)
        print(f"  ✅ Email sent to {args.to}")
    except FileNotFoundError:
        print("  ⚠  sendmail not found. Pass --smtp-host to use an SMTP server.")
        print(f"\n  Preview of plain-text report:\n{'─'*60}")
        print(plain[:3000])
    except Exception as e:
        print(f"  ✗ Failed to send: {e}")
        print(f"\n  Plain-text report:\n{'─'*60}")
        print(plain)


def torch_gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0).replace(" ", "_")
    except Exception:
        return "GPU"


if __name__ == "__main__":
    main()
