# AGENTS.md — letting an agent install the panel

Instructions for an AI agent (coding agent, nanobot, a Codex-like runner) that installs
this panel on the user's machine by itself. Humans: see [README.md](README.md).

Three rules that matter more than speed:

1. **No sudo, no `su`.** The panel installs under your own user. If a step needs root,
   stop and hand the user one command — do not look for a way around.
2. **Never expose the port.** `--host 127.0.0.1` is not negotiable; remote access is an
   ssh tunnel, and that is the user's decision, not yours.
3. **Do not touch other people's things.** ComfyUI and ollama are for using only: do not
   reinstall them, do not delete models, do not edit their configs. The panel writes
   inside its `PREFIX` and `panel.env`, nothing else.

## 1. Recon (read-only)

```bash
python3 -V                                  # need 3.10+
command -v ffmpeg ffprobe || echo "no ffmpeg — video tabs will not work"
curl -s -o /dev/null -w 'comfy %{http_code}\n' --max-time 5 http://127.0.0.1:8188/system_stats
curl -s -o /dev/null -w 'ollama %{http_code}\n' --max-time 5 http://127.0.0.1:11434/api/tags
python3 -c 'import psutil' 2>/dev/null && echo "psutil present" || echo "no psutil"
systemctl --user show-environment >/dev/null 2>&1 && echo "user systemd present" || echo "no user systemd"
```

What to do with the results:

- `comfy` is not 200 — the panel will install but cannot draw. Find where ComfyUI lives
  (`ss -ltn | grep 818`, `ps -ef | grep main.py`) and write its address into `COMFY_API`.
  Do not start ComfyUI for the user unless they asked.
- `ollama` is not 200 — translation and chat will not work. Ask which translator model
  the user has and put it into `UI_TRANSLATOR`.
- no `psutil` — the hardware tab will be empty. Offer it, do not install silently:
  `python3 -m pip install --user psutil`.
- no user systemd — `install.sh` falls back to a plain process with a log at
  `PREFIX/stand-ui.log`. That is fine; tell the user there will be no autostart.

## 2. Install

```bash
git clone https://github.com/piroman99/stand-panel && cd stand-panel
bash install.sh
```

The script copies `stand-ui.py` and `stand_guard.py` into `PREFIX` (default
`~/panels/stand-panel`), checks the syntax, creates `panel.env` from the example, installs
the user unit `stand-ui.service`, restarts it and calls the API to verify.

Another directory and port: `PREFIX=... PORT=... bash install.sh`.

## 3. Edit the config

`PREFIX/panel.env` is plain `NAME=value`, no quotes. Minimum worth setting:

```
COMFY_API=http://127.0.0.1:8188
OLLAMA_API=http://127.0.0.1:11434
COMFY_HOME=/path/to/ComfyUI        # output/, input/, models/loras/ are relative to it
UI_TRANSLATOR=<translator-model>  # for example qwen3:8b
UI_DISKS=/home                    # disks to show on the hardware tab
```

If the user has a separate models disk, add `UI_SATA_MOUNT=/mount/point` and
`UI_SATA_DEVICE=/sys/block/sdX`.

You can check that ComfyUI sees the right directories without guessing:
`curl -s http://127.0.0.1:8188/system_stats`.

After editing: `bash install.sh restart`.

## 4. Acceptance (do not skip)

```bash
bash install.sh status
```

Expected:

- `юнит: active` / unit active;
- port `8288` listening on **127.0.0.1 only**;
- `GET /` → `HTTP 200`;
- `GET /api/config` → non-empty `models`, `sizes`, `video_models`;
- `GET /api/sys` → `cpu`, `mem`, `gpu` (when there is a GPU).

Useful checks for the report to the user:

```bash
curl -s http://127.0.0.1:8288/api/config | python3 -c 'import json,sys; j=json.load(sys.stdin); print("models:", [m["key"] for m in j.get("models",[])]); print("video:", [m["key"] for m in j.get("video_models",[])])'
curl -s http://127.0.0.1:8288/api/sys | python3 -c 'import json,sys; j=json.load(sys.stdin); c=j.get("cpu") or {}; print("cpu", c.get("percent"), "temp", c.get("temp"), "watts", c.get("watts")); print("gpu", (j.get("gpu") or {}).get("name"))'
```

If `models` is empty the panel found no checkpoints: check `COMFY_HOME`/`COMFY_LORAS_DIR`
and that ComfyUI answers on `/object_info/CheckpointLoaderSimple`.

## 5. What you must not do

- run the panel as root or through `sudo`;
- change `--host 127.0.0.1` to `0.0.0.0`, open the port in a firewall, or put a reverse
  proxy in front "for convenience" — the panel has no authentication;
- invent models and paths: the model list comes from ComfyUI, the ollama model list from
  `/api/tags`, never from your head;
- promise tabs that are not in this repository (LoRA training, video upscaling, document
  base) — they need external scripts and images.

## 6. CPU watts (the only step that needs root)

The RAPL energy counter is root-only by default. The panel works without it, it just does
not show watts. If the user wants them, hand them one command to run themselves:

```bash
sudo chmod 444 /sys/class/powercap/intel-rapl:0/energy_uj
```

Warn that permissions reset on reboot; making it permanent needs a udev rule or
systemd-tmpfiles — that is their decision, not yours.

## 7. Removal

```bash
bash install.sh remove     # stop and remove the unit; code and panel.env stay
```

Tell the user exactly what remains on disk (`PREFIX`) before deleting anything.
