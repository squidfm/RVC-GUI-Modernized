import numpy as np, parselmouth, torch, pdb
from time import time as ttime
import torch.nn.functional as F
import torchcrepe # Fork feature. Use the crepe f0 algorithm. New dependency (pip install torchcrepe)
import scipy.signal as signal
import pyworld, os, traceback, faiss
from scipy import signal
from torch import Tensor # Fork Feature. Used for pitch prediction for the torchcrepe f0 inference computation

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

RMVPE_PATH = "rmvpe.pt"
RMVPE_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"


def ensure_rmvpe_model(path=RMVPE_PATH):
    """Download rmvpe.pt (~180 MB) on first use if it is not already there."""
    if os.path.isfile(path) and os.path.getsize(path) > 100 * 1024 * 1024:
        return path
    import requests
    print("rmvpe.pt not found - downloading it once from Hugging Face (~180 MB)...")
    tmp_path = path + ".part"
    with requests.get(RMVPE_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print("\r  %d / %d MB" % (done >> 20, total >> 20), end="")
    print()
    os.replace(tmp_path, path)
    return path


# Max quality preset: (x_pad, x_query, x_center, x_max) in seconds, and index clusters searched.
# x_pad = context on each side of a chunk, x_center = chunk length, x_max = files shorter than
# this are converted in one piece. Sized for a 12 GB GPU in fp32.
MAX_QUALITY_CHUNKS = (4, 10, 90, 100)
MAX_QUALITY_NPROBE = 64

_RMVPE_CACHE = {}  # (device, is_half) -> RMVPE model, so it loads once per session
_INDEX_CACHE = {}  # index path -> (faiss index, reconstructed vectors)


def get_rmvpe(device, is_half):
    key = (str(device), bool(is_half))
    if key not in _RMVPE_CACHE:
        from rmvpe import RMVPE
        print("loading rmvpe model (%s, %s)" % (device, "fp16" if is_half else "fp32"))
        _RMVPE_CACHE.clear()  # keep only one copy in VRAM
        _RMVPE_CACHE[key] = RMVPE(ensure_rmvpe_model(), is_half=is_half, device=device)
    return _RMVPE_CACHE[key]


def load_index(path):
    if path not in _INDEX_CACHE:
        print("loading index %s" % path)
        index = faiss.read_index(path)
        big_npy = index.reconstruct_n(0, index.ntotal)
        _INDEX_CACHE.clear()
        _INDEX_CACHE[path] = (index, big_npy)
    return _INDEX_CACHE[path]


def set_nprobe(index, nprobe):
    """IVF indexes only search `nprobe` of their clusters; RVC saves them with 1 (fast, approximate)."""
    try:
        ivf = faiss.extract_index_ivf(index)
        ivf.nprobe = int(min(max(1, nprobe), ivf.nlist))
    except Exception:
        pass  # flat (non-IVF) index: always exact


def change_rms(data1, sr1, data2, sr2, rate):
    """Blend the output's loudness envelope with the input's (from upstream RVC).
    rate = 1 keeps the output envelope, 0 copies the input envelope fully."""
    import librosa
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = F.interpolate(torch.from_numpy(rms1).unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = F.interpolate(torch.from_numpy(rms2).unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 = data2 * (torch.pow(rms1, torch.tensor(1 - rate)) * torch.pow(rms2, torch.tensor(rate - 1))).numpy()
    return data2


def combine_f0(f0_main, f0_second, agree_cents=50):
    """Merge two pitch tracks. Voicing comes from f0_main (RMVPE). Where both agree within
    `agree_cents` they are averaged (less jitter); where they disagree, the one closer to the
    local pitch trend wins (catches octave slips in either tracker)."""
    n = len(f0_main)
    f0_second = np.asarray(f0_second, dtype=np.float64)[:n]
    if len(f0_second) < n:
        f0_second = np.pad(f0_second, (0, n - len(f0_second)))
    out = np.array(f0_main, dtype=np.float64, copy=True)
    voiced = out > 0
    if voiced.sum() < 3:
        return out
    # local trend: main track with unvoiced gaps bridged, then median-filtered (~0.2 s)
    idx = np.arange(n)
    bridged = np.interp(idx, idx[voiced], np.log2(out[voiced]))
    trend = signal.medfilt(bridged, 21)
    both = voiced & (f0_second > 0)
    lm = np.zeros(n)
    ls = np.zeros(n)
    lm[both] = np.log2(out[both])
    ls[both] = np.log2(f0_second[both])
    diff = np.abs(lm - ls) * 1200
    agree = both & (diff < agree_cents)
    out[agree] = 2 ** ((lm[agree] + ls[agree]) / 2)
    disagree = both & ~agree
    pick_second = disagree & (np.abs(ls - trend) < np.abs(lm - trend))
    out[pick_second] = f0_second[pick_second]
    return out


def fix_octave_errors(f0, mode="down", jump=0.5, threshold=0.7):
    """Repair stretches where the pitch tracker locked onto half (or double) the real pitch.

    f0 is in Hz with 0 = unvoiced, 100 frames per second.
    The track is cut into pieces at unvoiced gaps and at abrupt jumps (> `jump` octaves
    between neighbouring frames). Any piece whose median sits more than `threshold`
    octaves below the median of the whole line is shifted up by whole octaves
    (mode "down"); mode "both" also pulls pieces that are too high back down.
    Returns (fixed_f0, number_of_frames_changed).
    """
    if mode in (None, "", "off"):
        return f0, 0
    f0 = np.array(f0, dtype=np.float64, copy=True)
    voiced = f0 > 0
    if voiced.sum() < 10:
        return f0, 0
    lf = np.zeros_like(f0)
    lf[voiced] = np.log2(f0[voiced])
    ref = np.median(lf[voiced])

    segments, start = [], None
    for i in range(len(f0)):
        if not voiced[i]:
            if start is not None:
                segments.append((start, i))
                start = None
            continue
        if start is None:
            start = i
        elif abs(lf[i] - lf[i - 1]) > jump:
            segments.append((start, i))
            start = i
    if start is not None:
        segments.append((start, len(f0)))

    changed = 0
    for s, e in segments:
        dev = np.median(lf[s:e]) - ref
        shift = 0
        if dev < -threshold:
            shift = int(np.round(-dev))  # move up this many octaves
        elif mode == "both" and dev > threshold:
            shift = -int(np.round(dev))  # move down
        if shift != 0:
            f0[s:e] *= 2.0 ** shift
            changed += e - s
    return f0, changed

class VC(object):
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.tgt_sr = tgt_sr
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * self.x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # 查询切点前后查询时间
        self.t_center = self.sr * self.x_center  # 查询切点位置
        self.t_max = self.sr * self.x_max  # 免查询时长阈值
        self.device = config.device
        # Pitch-tracking settings (can be changed from the GUI / batch script)
        self.f0_min = 50
        self.f0_max = 1100
        self.octave_fix = "down"  # "off", "down" or "both"
        self.last_octave_fixes = 0
        # Quality settings
        self.index_nprobe = 1      # clusters searched in the .index (1 = RVC default, fast)
        self.protect = 0.33        # <0.5: keep breaths/consonants from the source (upstream default 0.33)
        self.rms_mix_rate = 1.0    # 1 = output loudness envelope (off); lower = follow the input's dynamics
        self.crepe_batch = 512

    def get_f0_rmvpe(self, x, f0_min, f0_max, p_len):
        model = get_rmvpe(self.device, self.is_half)
        f0 = model.infer_from_audio(x.astype(np.float32), thred=0.03)
        f0[(f0 < f0_min) | (f0 > f0_max)] = 0
        if len(f0) < p_len:
            f0 = np.pad(f0, (0, p_len - len(f0)))
        return f0[:p_len] if len(f0) > p_len else f0

    #region f0 Overhaul Region
    # Fork Feature: Get the best torch device to use for f0 algorithms that require a torch device. Will return the type (torch.device)
    def get_optimal_torch_device(self, index: int = 0) -> torch.device:
        # Get cuda device
        if torch.cuda.is_available():
            return torch.device(f"cuda:{index % torch.cuda.device_count()}") # Very fast
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        # Insert an else here to grab "xla" devices if available. TO DO later. Requires the torch_xla.core.xla_model library
        # Else wise return the "cpu" as a torch device, 
        return torch.device("cpu")

    # Get the f0 via parselmouth computation
    def get_f0_pm_computation(self, x, time_step, f0_min, f0_max, p_len):
        f0 = (
            parselmouth.Sound(x, self.sr)
            .to_pitch_ac(
                time_step=time_step / 1000,
                voicing_threshold=0.6,
                pitch_floor=f0_min,
                pitch_ceiling=f0_max,
            )
            .selected_array["frequency"]
        )
        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.pad(
                f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
            )
        return f0

    # Get the f0 via the pyworld computation. Fork Feature +dio along with harvest
    def get_f0_pyworld_computation(self, x, f0_min, f0_max, f0_type):
        if f0_type == "harvest":
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=self.sr,
                f0_ceil=f0_max,
                f0_floor=f0_min,
                frame_period=10,
            )
        elif f0_type == "dio":
            f0, t = pyworld.dio(
                x.astype(np.double),
                fs=self.sr,
                f0_ceil=f0_max,
                f0_floor=f0_min,
                frame_period=10,
            )
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sr)
        f0 = signal.medfilt(f0, 3) 
        return f0
    
    # Fork Feature: Get the f0 via the crepe algorithm from torchcrepe
    def get_f0_crepe_computation(
            self, 
            x, 
            f0_min,
            f0_max,
            p_len,
            hop_length=128, # 512 before. Hop length changes the speed that the voice jumps to a different dramatic pitch. Lower hop lengths means more pitch accuracy but longer inference time.
            model="full", # Either use crepe-tiny "tiny" or crepe "full". Default is full
    ):
        x = x.astype(np.float32) # fixes the F.conv2D exception. We needed to convert double to float.
        x /= np.quantile(np.abs(x), 0.999)
        torch_device = self.get_optimal_torch_device()
        audio = torch.from_numpy(x).to(torch_device, copy=True)
        audio = torch.unsqueeze(audio, dim=0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True).detach()
        audio = audio.detach()
        print("Initiating prediction with a crepe_hop_length of: " + str(hop_length))
        pitch: Tensor = torchcrepe.predict(
            audio,
            self.sr,
            hop_length,
            f0_min,
            f0_max,
            model,
            batch_size=max(hop_length * 2, self.crepe_batch),
            device=torch_device,
            pad=True
        )
        p_len = p_len or x.shape[0] // hop_length
        # Resize the pitch for final f0
        source = np.array(pitch.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source
        )
        f0 = np.nan_to_num(target)
        return f0 # Resized f0
    
    #endregion

    def get_f0(self, x, p_len, f0_up_key, f0_method, crepe_hop_length, inp_f0=None):
        time_step = self.window / self.sr * 1000
        # Tracker search range comes from the settings; the mel scaling below must stay
        # at the fixed 50-1100 Hz range the models were trained with.
        f0_min = self.f0_min
        f0_max = self.f0_max
        f0_mel_min = 1127 * np.log(1 + 50 / 700)
        f0_mel_max = 1127 * np.log(1 + 1100 / 700)
        if f0_method == "rmvpe":
            f0 = self.get_f0_rmvpe(x, f0_min, f0_max, p_len)
        elif f0_method == "hybrid":  # RMVPE + crepe-full (fine hop), merged
            f0_r = self.get_f0_rmvpe(x, f0_min, f0_max, p_len)
            f0_c = self.get_f0_crepe_computation(x, f0_min, f0_max, p_len, 64, "full")
            f0 = combine_f0(f0_r, f0_c)
        elif f0_method == "pm":
            f0 = self.get_f0_pm_computation(x, time_step, f0_min, f0_max, p_len)
        elif f0_method == "harvest":
            f0 = self.get_f0_pyworld_computation(x, f0_min, f0_max, "harvest")
        elif f0_method == "dio": # Fork Feature
            f0 = self.get_f0_pyworld_computation(x, f0_min, f0_max, "dio")
        elif f0_method == "crepe": # Fork Feature: Adding a new f0 algorithm called crepe
            f0 = self.get_f0_crepe_computation(x, f0_min, f0_max, p_len, crepe_hop_length)
        elif f0_method == "crepe-tiny": # For Feature add crepe-tiny model
            f0 = self.get_f0_crepe_computation(x, f0_min, f0_max, p_len, crepe_hop_length, "tiny")

        print("Using the following f0 method: " + f0_method)
        f0, self.last_octave_fixes = fix_octave_errors(f0, self.octave_fix)
        if self.last_octave_fixes:
            print(
                "Octave guard: corrected %d pitch frames (%.2f s) that had dropped/jumped an octave"
                % (self.last_octave_fixes, self.last_octave_fixes / 100.0)
            )
        f0 *= pow(2, f0_up_key / 12)
        # with open("test.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        tf0 = self.sr // self.window  # 每秒f0点数
        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[
                :shape
            ]
        # with open("test_opt.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(int)

        return f0_coarse, f0bak  # 1-0

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        big_npy,
        index_rate,
        version,
    ):  # ,file_index,file_big_npy
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        use_protect = self.protect < 0.5 and pitch is not None and pitchf is not None
        if use_protect:
            feats0 = feats.clone()  # unindexed source features, used for breaths/consonants

        if (
            isinstance(index, type(None)) == False
            and isinstance(big_npy, type(None)) == False
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            # _, I = index.search(npy, 1)
            # npy = big_npy[I.squeeze()]

            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if use_protect:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        t1 = ttime()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch != None and pitchf != None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
        if use_protect:
            # voiced frames use the converted features; unvoiced frames (breaths,
            # s/t/k sounds) lean towards the source features so they don't turn metallic
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = self.protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            if pitch != None and pitchf != None:
                audio1 = (
                    (net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )
            else:
                audio1 = (
                    (net_g.infer(feats, p_len, sid)[0][0, 0]).data.cpu().float().numpy()
                )
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        times,
        f0_up_key,
        f0_method,
        file_index,
        # file_big_npy,
        index_rate,
        if_f0,
        version,
        crepe_hop_length,
        f0_file=None,
    ):
        if (
            file_index != ""
            # and file_big_npy != ""
            # and os.path.exists(file_big_npy) == True
            and os.path.exists(file_index) == True
            and index_rate != 0
        ):
            try:
                index, big_npy = load_index(file_index)
                set_nprobe(index, self.index_nprobe)
            except:
                traceback.print_exc()
                index = big_npy = None
        else:
            index = big_npy = None
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if hasattr(f0_file, "name") == True:
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
            except:
                traceback.print_exc()
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(audio_pad, p_len, f0_up_key, f0_method, crepe_hop_length, inp_f0)
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device, dtype=torch.float32).unsqueeze(0).float()
        t2 = ttime()
        times[1] += t2 - t1
        for t in opt_ts:
            t = t // self.window * self.window
            if if_f0 == 1:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        times,
                        index,
                        big_npy,
                        index_rate,
                         version,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
        if if_f0 == 1:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    times,
                    index,
                    big_npy,
                    index_rate,
                     version,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    times,
                    index,
                    big_npy,
                    index_rate,
                     version,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)
        if self.rms_mix_rate < 1:
            audio_opt = change_rms(audio, 16000, audio_opt, self.tgt_sr, self.rms_mix_rate)
        # avoid hard clipping when the file is written as 16-bit
        peak = np.abs(audio_opt).max() if len(audio_opt) else 0
        if peak > 0.99:
            audio_opt = audio_opt * (0.99 / peak)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt