"""Batch voice conversion for RVC-GUI.

Converts every audio file in a folder (subfolders included) with one RVC model and
writes .wav files with the same names/structure to an output folder. Uses the same
pipeline as the GUI, including the RMVPE pitch method and the octave guard.

Examples (run from the RVC-GUI-pkg folder, or use RVC-Batch.bat):
    runtime\\python.exe rvc_batch.py --list-models
    runtime\\python.exe rvc_batch.py --model HeavyTF2V2_655e_32750s --input D:\\lines --pitch -3
"""
import argparse
import csv
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)  # ffmpeg.exe, hubert_base.pt, rmvpe.pt and models/ are looked up here
sys.path.insert(0, HERE)

AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".wma", ".opus", ".webm"}
MODELS_DIR = os.path.join(HERE, "models")


def parse_args():
    ap = argparse.ArgumentParser(description="Batch RVC voice conversion")
    ap.add_argument("--model", help="model folder name inside models\\, a folder path, or a .pth path")
    ap.add_argument("--input", help="folder (or single file) with the source voice lines")
    ap.add_argument("--output", help="output folder (default: output\\<model name>)")
    ap.add_argument("--pitch", type=int, default=0, help="transpose in semitones (e.g. 12 = one octave up)")
    ap.add_argument("--method", default="rmvpe",
                    choices=["rmvpe", "hybrid", "harvest", "crepe", "crepe-tiny", "dio", "pm"],
                    help="pitch tracking method (default rmvpe; max quality turns rmvpe into hybrid)")
    ap.add_argument("--quality", default="max", choices=["max", "normal"],
                    help="max = fp32, RMVPE+crepe pitch, wider index search, bigger chunks (slower)")
    ap.add_argument("--protect", type=float, default=0.33,
                    help="protect consonants/breaths, 0-0.5 (0.5 = off, default 0.33)")
    ap.add_argument("--rms-mix", type=float, default=1.0,
                    help="volume envelope mix 0-1 (1 = off, lower follows the input's loudness)")
    ap.add_argument("--nprobe", type=int, help="index clusters to search (default: 64 in max quality, 1 normal)")
    ap.add_argument("--index", help=".index file (default: picked from the model folder)")
    ap.add_argument("--index-rate", type=float, default=0.4, help="feature retrieval rate 0-1 (GUI default 0.4)")
    ap.add_argument("--octave-fix", default="down", choices=["down", "both", "off"],
                    help="repair octave drops (down), drops and jumps (both), or nothing (off)")
    ap.add_argument("--f0-min", type=float, default=50, help="lowest pitch the tracker may report, Hz")
    ap.add_argument("--f0-max", type=float, default=1100, help="highest pitch the tracker may report, Hz")
    ap.add_argument("--crepe-hop", type=int, default=128, help="hop length for crepe methods")
    ap.add_argument("--sr", default="model",
                    help="output sample rate: 'model' (default), 'source' (match each input file) or a number")
    ap.add_argument("--overwrite", action="store_true", help="re-convert files that already exist in the output")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    ap.add_argument("--list-models", action="store_true", help="show the available models and exit")
    return ap.parse_args()


def list_models():
    out = []
    if not os.path.isdir(MODELS_DIR):
        return out
    for name in sorted(os.listdir(MODELS_DIR)):
        d = os.path.join(MODELS_DIR, name)
        if os.path.isdir(d):
            pths = [f for f in os.listdir(d) if f.endswith(".pth") and not f.startswith(("G_", "D_"))]
            if pths:
                out.append((name, pths))
    return out


def resolve_model(model_arg, index_arg):
    """Return (pth_path, index_path_or_empty, display_name)."""
    if model_arg.lower().endswith(".pth"):
        pth = model_arg if os.path.isabs(model_arg) else os.path.join(MODELS_DIR, model_arg)
        if not os.path.isfile(pth):
            pth = os.path.abspath(model_arg)
        model_dir = os.path.dirname(pth)
    else:
        model_dir = model_arg if os.path.isdir(model_arg) else os.path.join(MODELS_DIR, model_arg)
        if not os.path.isdir(model_dir):
            raise SystemExit("Model '%s' not found. Use --list-models to see the names." % model_arg)
        pths = [f for f in os.listdir(model_dir) if f.endswith(".pth")
                and not f.startswith(("G_", "D_"))
                and os.path.getsize(os.path.join(model_dir, f)) < 200 * 1024 ** 2]
        if not pths:
            raise SystemExit("No usable .pth file in %s" % model_dir)
        if len(pths) > 1:
            raise SystemExit("%s has several voices, pick one with --model \"%s\\<file>.pth\":\n  %s"
                             % (model_dir, os.path.basename(model_dir), "\n  ".join(sorted(pths))))
        pth = os.path.join(model_dir, pths[0])
    if not os.path.isfile(pth):
        raise SystemExit("Model file not found: %s" % pth)

    index = ""
    if index_arg:
        index = index_arg
    else:
        idx = [f for f in os.listdir(model_dir) if f.endswith(".index") and not f.startswith("trained")]
        stem = os.path.splitext(os.path.basename(pth))[0].lower()
        matching = [f for f in idx if stem in f.lower()]
        if len(idx) == 1:
            index = os.path.join(model_dir, idx[0])
        elif len(matching) == 1:
            index = os.path.join(model_dir, matching[0])
        elif idx:
            print("Several .index files in %s and none clearly matches; running without index "
                  "(pass --index to choose one)." % model_dir)
    name = os.path.splitext(os.path.basename(pth))[0]
    return pth, index, name


def collect_inputs(inp):
    if os.path.isfile(inp):
        return os.path.dirname(os.path.abspath(inp)), [os.path.abspath(inp)]
    files = []
    for root, _, names in os.walk(inp):
        for n in sorted(names):
            if os.path.splitext(n)[1].lower() in AUDIO_EXTS:
                files.append(os.path.join(root, n))
    return os.path.abspath(inp), files


def source_samplerate(path):
    try:
        import soundfile as sf
        return sf.info(path).samplerate
    except Exception:
        import ffmpeg
        info = ffmpeg.probe(path, cmd="ffprobe") if os.path.exists("ffprobe.exe") else None
        if info:
            for s in info.get("streams", []):
                if s.get("codec_type") == "audio":
                    return int(s["sample_rate"])
    return None


def main():
    args = parse_args()

    if args.list_models or not args.model:
        models = list_models()
        print("Available models (folder name -> voice files):")
        for name, pths in models:
            print("  %-40s %s" % (name, ", ".join(pths)))
        if not args.list_models:
            print("\nNo --model given. Set MODEL in RVC-Batch.bat or pass --model <name>.")
        return
    if not args.input:
        raise SystemExit("No --input folder given.")
    if not os.path.exists(args.input):
        raise SystemExit("Input not found: %s" % args.input)

    pth, index, model_name = resolve_model(args.model, args.index)
    in_root, files = collect_inputs(args.input)
    if not files:
        raise SystemExit("No audio files found in %s" % args.input)
    out_root = os.path.abspath(args.output or os.path.join(HERE, "output", model_name))
    os.makedirs(out_root, exist_ok=True)

    # config.py parses the command line itself, so hide our arguments from it
    sys.argv = [sys.argv[0]] + (["--use_gfloat"] if args.cpu else [])
    tmp = os.path.join(HERE, "TEMP")
    os.makedirs(tmp, exist_ok=True)
    os.environ["TEMP"] = tmp
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import soundfile as sf
    import torch
    from config import Config
    from fairseq import checkpoint_utils
    from my_utils import load_audio
    from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
    from infer_pack.modelsv2 import SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono
    from vc_infer_pipeline import VC, MAX_QUALITY_CHUNKS, MAX_QUALITY_NPROBE

    torch.manual_seed(114514)
    config = Config()
    max_q = args.quality == "max"
    if args.cpu:
        config.device = "cpu"
        config.is_half = False
        config.x_pad, config.x_query, config.x_center, config.x_max = 1, 6, 38, 41
    if max_q:
        config.is_half = False
        if not args.cpu:
            config.x_pad, config.x_query, config.x_center, config.x_max = MAX_QUALITY_CHUNKS
        if args.method == "rmvpe":
            args.method = "hybrid"
        if args.method in ("crepe", "crepe-tiny"):
            args.crepe_hop = min(args.crepe_hop, 64)
    nprobe = args.nprobe if args.nprobe else (MAX_QUALITY_NPROBE if max_q else 1)
    print("Device: %s  (half precision: %s)" % (config.device, config.is_half))

    print("Loading HuBERT...")
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"], suffix="")
    hubert = models[0].to(config.device)
    hubert = hubert.half() if config.is_half else hubert.float()
    hubert.eval()

    print("Loading voice model: %s" % pth)
    cpt = torch.load(pth, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        net_g = (SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half) if if_f0 == 1
                 else SynthesizerTrnMs256NSFsid_nono(*cpt["config"]))
    else:
        net_g = (SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half) if if_f0 == 1
                 else SynthesizerTrnMs768NSFsid_nono(*cpt["config"]))
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to(config.device)
    net_g = net_g.half() if config.is_half else net_g.float()

    vc = VC(tgt_sr, config)
    vc.f0_min, vc.f0_max = args.f0_min, args.f0_max
    vc.octave_fix = args.octave_fix
    vc.index_nprobe = nprobe
    vc.protect = args.protect
    vc.rms_mix_rate = args.rms_mix

    print("Model: %s (%s, %d Hz, pitch-guided: %s)" % (model_name, version, tgt_sr, bool(if_f0)))
    print("Index: %s" % (index or "none"))
    print("Quality: %s (%s, chunks %ds + %ds context) | index nprobe: %d"
          % (args.quality, "fp16" if config.is_half else "fp32", config.x_center, config.x_pad, nprobe))
    print("Method: %s | transpose: %+d | index rate: %.2f | protect: %.2f | volume mix: %.2f | octave guard: %s"
          % (args.method, args.pitch, args.index_rate, args.protect, args.rms_mix, args.octave_fix))
    print("Converting %d file(s)\n  from %s\n  to   %s\n" % (len(files), in_root, out_root))

    log_rows = []
    n_ok = n_skip = n_fail = 0
    flagged = []
    t_start = time.time()
    for i, src in enumerate(files, 1):
        rel = os.path.relpath(src, in_root)
        dst = os.path.join(out_root, os.path.splitext(rel)[0] + ".wav")
        if os.path.exists(dst) and not args.overwrite:
            n_skip += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print("[%d/%d] %s" % (i, len(files), rel))
        try:
            audio = load_audio(src, 16000)
            dur = len(audio) / 16000.0
            times = [0, 0, 0]
            vc.last_octave_fixes = 0
            out = vc.pipeline(hubert, net_g, 0, audio, times, args.pitch, args.method,
                              index, args.index_rate, if_f0, version, args.crepe_hop, None)
            out_sr = tgt_sr
            if args.sr != "model":
                want = source_samplerate(src) if args.sr == "source" else int(args.sr)
                if want and want != tgt_sr:
                    import librosa
                    out = librosa.resample(out.astype(np.float32), orig_sr=tgt_sr, target_sr=want)
                    out_sr = want
            sf.write(dst, out, out_sr, subtype="PCM_16")
            fixes = vc.last_octave_fixes
            if fixes:
                flagged.append((rel, fixes))
            log_rows.append([rel, "ok", "%.2f" % dur, "%.2f" % (fixes / 100.0), ""])
            n_ok += 1
        except Exception as e:
            traceback.print_exc()
            log_rows.append([rel, "FAILED", "", "", str(e).splitlines()[0] if str(e) else repr(e)])
            n_fail += 1

    log_path = os.path.join(out_root, "batch_log.csv")
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["file", "status", "seconds", "octave_fixed_seconds", "error"])
        w.writerows(log_rows)

    print("\nDone in %.0f s: %d converted, %d skipped (already there), %d failed."
          % (time.time() - t_start, n_ok, n_skip, n_fail))
    if flagged:
        print("\nThe octave guard repaired pitch in these files - worth a quick listen:")
        for rel, fixes in flagged:
            print("  %-50s %.2f s corrected" % (rel, fixes / 100.0))
    print("Log: %s" % log_path)


if __name__ == "__main__":
    main()
