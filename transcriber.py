# Транскрибатор – быстрый перевод аудио/видео в текст (faster-whisper, GPU).
# Выбор модели, файлы вручную или слежение за папкой, оценка времени,
# перенос исходника и результатов в папку проекта, трей, автозапуск.

import os, sys, json, time, threading, queue, shutil, ctypes, subprocess, traceback
from pathlib import Path

# ---------- пути и окружение ----------

def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent

BASE = app_dir()
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)  # модели всегда рядом с программой (portable)
os.environ.setdefault("HF_HOME", str(MODELS_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

CFG_PATH = BASE / "transcriber_config.json"
LOG_PATH = BASE / "transcriber.log"
PRIVATE_DIR = BASE / "private"
KEY_PATH = PRIVATE_DIR / "kimi_key"

# ----- защищённое хранение ключа Kimi Code (отдельно от конфига) -----

def save_key(key: str):
    """Атомарно записать ключ; доступ только у текущего пользователя."""
    PRIVATE_DIR.mkdir(exist_ok=True)
    tmp = KEY_PATH.with_suffix(".tmp")
    tmp.write_text(key.strip(), encoding="utf-8")
    os.replace(tmp, KEY_PATH)
    try:  # наследование прав off, полный доступ только владельцу
        subprocess.run(["icacls", str(KEY_PATH), "/inheritance:r", "/grant:r",
                        f"{os.environ.get('USERDOMAIN', '')}\\{os.environ.get('USERNAME', '')}:F"],
                       capture_output=True, creationflags=0x08000000)  # CREATE_NO_WINDOW
    except Exception as e:
        log("icacls:", e)

def load_key() -> str:
    try:
        return KEY_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

def delete_key():
    try:
        KEY_PATH.unlink(missing_ok=True)
    except Exception:
        pass

def migrate_key(cfg: dict):
    """Перенести ключ из старого конфига в защищённое хранилище."""
    old = (cfg.get("ai_key") or "").strip()
    if old:
        if not load_key():
            try:
                save_key(old)
            except Exception as e:
                log("миграция ключа:", e)
        cfg.pop("ai_key", None)
        try:
            CFG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

MEDIA_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".wma", ".aac",
             ".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".3gp", ".ts"}

MODEL_LIST = ["large-v3-turbo", "large-v3", "distil-large-v3", "medium", "small", "base", "tiny"]

# примерная скорость распознавания, «во сколько раз быстрее длительности записи»
# (после первого файла уточняется реальным замером на этом компьютере)
SPEED = {
    ("large-v3-turbo", "cuda"): 20.0, ("large-v3", "cuda"): 12.0,
    ("distil-large-v3", "cuda"): 28.0, ("medium", "cuda"): 45.0,
    ("small", "cuda"): 75.0, ("base", "cuda"): 130.0, ("tiny", "cuda"): 200.0,
    ("large-v3-turbo", "cpu"): 2.5, ("large-v3", "cpu"): 1.5,
    ("distil-large-v3", "cpu"): 3.0, ("medium", "cpu"): 5.0,
    ("small", "cpu"): 9.0, ("base", "cpu"): 15.0, ("tiny", "cpu"): 25.0,
}
OVERHEAD_S = 4.0  # накладные расходы на файл (открытие, VAD и т.п.)

def log(*a):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a) + "\n")
    except Exception:
        pass

def add_cuda_dlls():
    """Подсказать Windows, где лежат pip-установленные DLL CUDA (cuBLAS, cuDNN)."""
    roots = []
    if getattr(sys, "frozen", False):
        roots.append(Path(getattr(sys, "_MEIPASS", BASE)))
        roots.append(BASE / "_internal")
    roots.append(Path(sys.prefix) / "Lib" / "site-packages")
    roots.append(BASE)
    for r in roots:
        nv = r / "nvidia"
        if not nv.is_dir():
            continue
        for sub in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
            d = nv / sub / "bin"
            if d.is_dir():
                os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]
                try:
                    os.add_dll_directory(str(d))
                except Exception:
                    pass

def cuda_available() -> bool:
    try:
        add_cuda_dlls()
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False

def enable_dpi_awareness():
    """Чёткие шрифты на мониторах с масштабированием (2K/4K)."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

def dark_titlebar(root):
    """Тёмный заголовок окна (Windows 10 1809+ / 11)."""
    try:
        root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id()) or root.winfo_id()
        val = ctypes.c_int(1)
        for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(val), 4)
    except Exception as e:
        log("titlebar:", e)

def autostart_link() -> Path:
    return Path(os.environ["APPDATA"]) / r"Microsoft\Windows\Start Menu\Programs\Startup\Transcriber.lnk"

def set_autostart(enable: bool):
    """Ярлык в автозагрузке Windows."""
    lnk = autostart_link()
    try:
        if enable:
            target = str(Path(sys.executable).resolve())
            ps = ("$s=New-Object -ComObject WScript.Shell;"
                  f"$l=$s.CreateShortcut('{lnk}');$l.TargetPath='{target}';"
                  f"$l.WorkingDirectory='{BASE}';$l.Description='Транскрибатор';$l.Save()")
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True)
        else:
            lnk.unlink(missing_ok=True)
    except Exception as e:
        log("автозапуск:", e)

def probe_duration(path: Path):
    """Длительность файла в секундах через PyAV (ffmpeg не нужен)."""
    try:
        import av
        c = av.open(str(path))
        try:
            if c.duration:
                return c.duration / 1_000_000.0
        finally:
            c.close()
    except Exception as e:
        log("длительность не определена:", path.name, e)
    return None

def fmt_time(sec) -> str:
    if sec is None:
        return "—"
    sec = max(0, int(round(sec)))
    h, sec = divmod(sec, 3600)
    m, s = divmod(sec, 60)
    if h:
        return f"{h} ч {m:02d} мин"
    if m:
        return f"{m} мин {s:02d} с"
    return f"{s} с"

def fmt_dur(sec) -> str:
    if sec is None:
        return "?"
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def srt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def move_to_project(paths, project_dir) -> dict:
    """Перенести файлы в папку проекта; вернуть {старый путь: новый путь}."""
    dst_dir = Path(project_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    moved = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        dst = dst_dir / p.name
        if dst.exists():
            n = 1
            while (dst_dir / f"{p.stem}_{n}{p.suffix}").exists():
                n += 1
            dst = dst_dir / f"{p.stem}_{n}{p.suffix}"
        shutil.move(str(p), str(dst))
        moved[str(p)] = str(dst)
    return moved

def run_ai(command: str, prompt: str, text: str):
    """Отправить промпт + текст команде через stdin, вернуть ответ или None."""
    try:
        proc = subprocess.run(command, input=(prompt + "\n\n" + text) if prompt else text,
                              shell=True, capture_output=True, timeout=900,
                              encoding="utf-8", errors="replace")
        out = (proc.stdout or "").strip()
        if proc.returncode != 0:
            log("ИИ-команда завершилась с кодом", proc.returncode, (proc.stderr or "")[:500])
        return out or None
    except Exception as e:
        log("ИИ-обработка:", e)
        return None

KIMI_ERRORS = {
    "KIMI_API_INVALID_KEY": "ключ не задан или некорректен (Настройки → API-ключ)",
    "KIMI_API_AUTH_REQUIRED": "ключ отклонён (401) — проверьте ключ в Kimi Code Console",
    "KIMI_API_ACCESS_DENIED": "доступ запрещён (403) — у ключа нет доступа к этой модели",
    "KIMI_API_USAGE_LIMIT": "превышена квота (429) — попробуйте позже",
    "KIMI_API_BAD_REQUEST": "запрос отклонён (400) — проверьте модель и настройки",
    "KIMI_API_TIMEOUT": "истекло время ожидания ответа",
    "KIMI_API_UNAVAILABLE": "сервис недоступен — проверьте интернет и регион",
    "KIMI_API_INVALID_RESPONSE": "неожиданный ответ сервиса",
    "KIMI_API_INCOMPLETE_RESPONSE": "ответ оборвался (finish_reason != stop)",
    "KIMI_API_UNEXPECTED_TOOL_CALL": "модель вернула tool_call вместо текста",
}

def run_kimi_text(region: str, model: str, prompt: str, text: str):
    """Один вызов Kimi Code API: вернуть текст ответа или None. Ключ — из защищённого хранилища."""
    try:
        import kimi_code_api as kca
        key = load_key()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": ((prompt + "\n\n") if prompt else "") + text}],
            "stream": False, "max_tokens": 8192,
            "thinking": {"type": "enabled"},
        }
        reply = kca.http_json(region, key, "/chat/completions", payload, timeout=600)
        choices = reply.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("KIMI_API_INVALID_RESPONSE")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("KIMI_API_INVALID_RESPONSE")
        if message.get("tool_calls") or message.get("function_call"):
            raise ValueError("KIMI_API_UNEXPECTED_TOOL_CALL")
        if choice.get("finish_reason") != "stop":
            raise ValueError("KIMI_API_INCOMPLETE_RESPONSE")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("KIMI_API_INVALID_RESPONSE")
        usage = reply.get("usage")
        if isinstance(usage, dict):
            log("kimi usage:", usage.get("prompt_tokens"), "->", usage.get("completion_tokens"),
                "модель:", reply.get("model"))
        return content.strip()
    except ValueError as e:
        code = str(e)
        log("Kimi API:", code)
        raise RuntimeError(KIMI_ERRORS.get(code, "ошибка Kimi API: " + code)) from None
    except Exception as e:
        log("Kimi API:", e)
        return None

# ---------- ядро распознавания ----------

class Engine:
    def __init__(self):
        self.model = None
        self.key = None

    def ensure(self, model_name: str, device: str, status=print):
        dev = device
        if dev == "auto":
            dev = "cuda" if cuda_available() else "cpu"
        key = (model_name, dev)
        if self.model is not None and self.key == key:
            return dev
        status(f"Загрузка модели {model_name} ({'видеокарта' if dev == 'cuda' else 'процессор'})…")
        add_cuda_dlls()
        from faster_whisper import WhisperModel
        self.model = None  # освободить старую
        if dev == "cuda":
            try:
                self.model = WhisperModel(model_name, device="cuda", compute_type="float16")
            except Exception as e:
                log("CUDA не завелась, перехожу на процессор:", e)
                dev = "cpu"
        if self.model is None:
            threads = max(2, (os.cpu_count() or 4) - 1)
            self.model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=threads)
        self.key = (model_name, dev)
        status("Модель готова")
        return dev

    def transcribe(self, path: Path, language, progress=None, prompt=None):
        segs, info = self.model.transcribe(
            str(path), language=language, beam_size=5, vad_filter=True,
            initial_prompt=prompt or None, condition_on_previous_text=False)
        total = getattr(info, "duration", None)
        out = []
        for s in segs:
            out.append((s.start, s.end, s.text.strip()))
            if progress and total:
                progress(min(1.0, s.end / total))
        if progress:
            progress(1.0)
        return out

# ---------- самопроверка без GUI: transcriber.py --selftest файл.wav ----------

def selftest():
    wav = Path(sys.argv[sys.argv.index("--selftest") + 1])
    eng = Engine()
    dev = eng.ensure("large-v3-turbo", "auto")
    t0 = time.time()
    segs = eng.transcribe(wav, "ru")
    dt = time.time() - t0
    dur = probe_duration(wav)
    text = " ".join(s[2] for s in segs)
    res = (f"OK device={dev} duration={dur:.1f}s transcribe={dt:.1f}s\n{text}\n")
    (BASE / "selftest_result.txt").write_text(res, encoding="utf-8")
    print(res)

# ---------- тема оформления ----------

C = {
    "bg": "#1E2230", "panel": "#262B3D", "panel2": "#2E3448",
    "fg": "#E8EAF2", "muted": "#9AA3C0", "accent": "#567FE8",
    "accent2": "#7A9BEF", "green": "#3FB950", "red": "#F85149", "yellow": "#D29922",
}

def apply_theme(root):
    root.configure(bg=C["bg"])
    root.option_add("*TCombobox*Listbox.background", C["panel2"])
    root.option_add("*TCombobox*Listbox.foreground", C["fg"])
    root.option_add("*TCombobox*Listbox.selectBackground", C["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")
    from tkinter import ttk
    st = ttk.Style(root)
    st.theme_use("clam")
    st.configure(".", background=C["bg"], foreground=C["fg"],
                 fieldbackground=C["panel"], bordercolor=C["panel2"],
                 font=("Segoe UI", 10))
    st.configure("TFrame", background=C["bg"])
    st.configure("Card.TFrame", background=C["panel"])
    st.configure("TLabel", background=C["bg"], foreground=C["fg"])
    st.configure("Card.TLabel", background=C["panel"], foreground=C["fg"])
    st.configure("Muted.TLabel", background=C["bg"], foreground=C["muted"])
    st.configure("CardMuted.TLabel", background=C["panel"], foreground=C["muted"])
    st.configure("Title.TLabel", background=C["bg"], foreground=C["fg"],
                 font=("Segoe UI", 14, "bold"))
    st.configure("TButton", background=C["panel2"], foreground=C["fg"],
                 padding=(12, 6), borderwidth=0, focusthickness=0)
    st.map("TButton",
           background=[("active", C["accent"]), ("pressed", C["accent"]), ("disabled", C["panel"])],
           foreground=[("disabled", C["muted"])])
    st.configure("Accent.TButton", background=C["accent"], foreground="#FFFFFF",
                 padding=(14, 6), font=("Segoe UI", 10, "bold"))
    st.map("Accent.TButton",
           background=[("active", C["accent2"]), ("pressed", C["accent2"]), ("disabled", C["panel"])],
           foreground=[("disabled", C["muted"])])
    st.configure("TCheckbutton", background=C["bg"], foreground=C["fg"])
    st.map("TCheckbutton", background=[("active", C["bg"])])
    st.configure("Card.TCheckbutton", background=C["panel"], foreground=C["fg"])
    st.map("Card.TCheckbutton", background=[("active", C["panel"])])
    st.configure("TEntry", fieldbackground=C["panel2"], foreground=C["fg"],
                 insertcolor=C["fg"], bordercolor=C["panel2"])
    st.configure("TCombobox", fieldbackground=C["panel2"], background=C["panel2"],
                 foreground=C["fg"], arrowcolor=C["fg"], bordercolor=C["panel2"])
    st.map("TCombobox",
           fieldbackground=[("readonly", C["panel2"])],
           foreground=[("readonly", C["fg"])],
           selectbackground=[("readonly", C["panel2"])],
           selectforeground=[("readonly", C["fg"])])
    st.configure("Treeview", background=C["panel"], fieldbackground=C["panel"],
                 foreground=C["fg"], rowheight=28, borderwidth=0)
    st.configure("Treeview.Heading", background=C["panel2"], foreground=C["muted"],
                 font=("Segoe UI", 9, "bold"), borderwidth=0, padding=(6, 6))
    st.map("Treeview", background=[("selected", C["accent"])],
           foreground=[("selected", "#FFFFFF")])
    st.configure("accent.Horizontal.TProgressbar", background=C["accent"],
                 troughcolor=C["panel2"], borderwidth=0, thickness=10)
    st.configure("TMenubutton", background=C["bg"], foreground=C["fg"],
                 padding=(10, 2), font=("Segoe UI", 13), borderwidth=0)
    st.map("TMenubutton", background=[("active", C["panel2"])])

# ---------- графический интерфейс ----------

def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    cfg = {}
    if CFG_PATH.exists():
        try:
            cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    migrate_key(cfg)  # если ключ лежал в конфиге — увезти в защищённое хранилище

    root = tk.Tk()
    root.title("Транскрибатор — аудио/видео в текст")
    root.geometry("1040x700")
    root.minsize(880, 580)
    apply_theme(root)
    dark_titlebar(root)

    q = queue.Queue()          # события из рабочих потоков в GUI
    engine = Engine()
    state = {
        "busy": False, "stop": False, "watching": False,
        "watch_dir": cfg.get("watch_dir", ""), "seen": set(),
        "measured": cfg.get("measured", {}),  # "model|device" -> фактор скорости
        "results": {},           # iid -> путь к .txt
        "auto_batch": False,     # текущая порция запущена автоматически (слежение)
        "auto_items": [],        # файлы авто-порции: {"iid","name","files"}
        "cb_kmodel": None,       # combobox моделей в настройках (для обновления каталога)
    }

    v_model = tk.StringVar(value=cfg.get("model", "large-v3-turbo"))
    v_dev = tk.StringVar(value={"auto": "Авто", "cuda": "Видеокарта", "cpu": "Процессор"}.get(cfg.get("device", "auto"), "Авто"))
    v_lang = tk.StringVar(value={"ru": "Русский", "auto": "Авто", "en": "Английский"}.get(cfg.get("language", "ru"), "Русский"))
    v_srt = tk.BooleanVar(value=bool(cfg.get("srt", False)))
    v_struct = tk.BooleanVar(value=bool(cfg.get("structured", True)))
    v_auto = tk.BooleanVar(value=bool(cfg.get("auto_process", True)))
    v_autostart = tk.BooleanVar(value=autostart_link().exists())
    v_watch = tk.StringVar(value=state["watch_dir"])
    v_project = tk.StringVar(value=cfg.get("project_dir", ""))
    v_ai_on = tk.BooleanVar(value=bool(cfg.get("ai_enabled", False)))
    v_ai_cmd = tk.StringVar(value=cfg.get("ai_command", ""))
    v_ai_mode = tk.StringVar(value=cfg.get("ai_mode", "api"))       # api | cli
    v_ai_region = tk.StringVar(value=cfg.get("ai_region", "global"))  # global | mainland-cn
    v_ai_key_new = tk.StringVar(value="")   # поле ввода ключа: всегда пустое, ключ не показывается
    v_ai_model = tk.StringVar(value=cfg.get("ai_model", "kimi-for-coding"))

    KIMI_MODELS = ["kimi-for-coding", "kimi-for-coding-highspeed",
                   "k3.1-pro", "k3.1-pro-highspeed", "k3", "k3-256k"]

    DEFAULT_AI_PROMPT = ("Сделай структурированный конспект: основные тезисы, принятые решения, "
                         "поручения с ответственными. Отвечай по-русски.")
    AI_PRESETS = {"Kimi CLI": "kimi --print", "Ollama": "ollama run llama3.1", "Своя команда": ""}
    ai_prompt_text = {"v": cfg.get("ai_prompt", DEFAULT_AI_PROMPT)}
    whisper_prompt_text = {"v": cfg.get("whisper_prompt", "")}

    DEV_MAP = {"Авто": "auto", "Видеокарта": "cuda", "Процессор": "cpu"}
    LANG_MAP = {"Русский": "ru", "Авто": None, "Английский": "en"}

    def save_cfg(*_):
        try:
            CFG_PATH.write_text(json.dumps({
                "model": v_model.get(), "device": DEV_MAP[v_dev.get()],
                "language": v_lang.get() if v_lang.get() != "Авто" else "auto",
                "srt": v_srt.get(), "structured": v_struct.get(),
                "auto_process": v_auto.get(), "watch_dir": state["watch_dir"],
                "project_dir": v_project.get().strip(),
                "ai_enabled": v_ai_on.get(), "ai_command": v_ai_cmd.get(),
                "ai_mode": v_ai_mode.get(), "ai_region": v_ai_region.get(),
                "ai_model": v_ai_model.get().strip(),
                "ai_prompt": ai_prompt_text["v"], "whisper_prompt": whisper_prompt_text["v"],
                "measured": state["measured"],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def resolved_dev():
        d = DEV_MAP[v_dev.get()]
        if d == "auto":
            d = "cuda" if cuda_available() else "cpu"
        return d

    def estimate(model, dev, dur):
        if dur is None:
            return None
        key = f"{model}|{dev}"
        f = state["measured"].get(key) or SPEED.get((model, dev), 10.0)
        return dur / f + OVERHEAD_S

    def on_autostart_toggle():
        set_autostart(v_autostart.get())
        save_cfg()
        status("Автозапуск " + ("включён — программа стартует вместе с Windows" if v_autostart.get()
                                else "выключен"))

    def open_settings():
        win = tk.Toplevel(root)
        win.title("Настройки")
        win.configure(bg=C["bg"])
        win.transient(root)
        frame = ttk.Frame(win, style="Card.TFrame", padding=14)
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        ttk.Checkbutton(frame, text="Запускать программу вместе с Windows",
                        variable=v_autostart, style="Card.TCheckbutton",
                        command=on_autostart_toggle).pack(anchor="w")
        ttk.Separator(frame).pack(fill="x", pady=10)

        # --- подсказка для распознавания ---
        ttk.Label(frame, text="Распознавание (Whisper)",
                  style="Card.TLabel", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(frame, style="CardMuted.TLabel", wraplength=580,
                  text="Подсказка распознаванию — перечислите термины, названия сервисов и брендов "
                       "как они должны писаться (например: Kimi, Whisper, CUDA, Bitrix24). "
                       "Whisper подстроит написание под подсказку.").pack(anchor="w", pady=(4, 2))
        txt_whisper = tk.Text(frame, height=2, wrap="word", bg=C["panel2"], fg=C["fg"],
                              insertbackground=C["fg"], relief="flat", font=("Segoe UI", 10))
        txt_whisper.pack(fill="x", pady=(0, 4))
        txt_whisper.insert("1.0", whisper_prompt_text["v"])
        ttk.Separator(frame).pack(fill="x", pady=10)

        # --- нейросеть для постобработки ---
        ttk.Label(frame, text="Обработка результата нейросетью",
                  style="Card.TLabel", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Checkbutton(frame, text="после распознавания отправлять текст нейросети",
                        variable=v_ai_on, style="Card.TCheckbutton",
                        command=save_cfg).pack(anchor="w", pady=(6, 8))

        row = ttk.Frame(frame, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text="Способ:", style="Card.TLabel").pack(side="left")
        MODES = ["Kimi API (ключ)", "Kimi CLI", "Ollama", "Своя команда"]
        cb_mode = ttk.Combobox(row, values=MODES, width=18, state="readonly")
        cb_mode.pack(side="left", padx=(6, 0))
        cb_mode.set("Kimi API (ключ)" if v_ai_mode.get() == "api"
                    else ("Kimi CLI" if "kimi" in v_ai_cmd.get()
                          else "Ollama" if "ollama" in v_ai_cmd.get() else "Своя команда"))

        api_frame = ttk.Frame(frame, style="Card.TFrame")
        ttk.Label(api_frame, text="Регион:", style="CardMuted.TLabel").grid(row=0, column=0, sticky="w")
        cb_region = ttk.Combobox(api_frame, textvariable=v_ai_region, state="readonly",
                                 values=["global", "mainland-cn"], width=15)
        cb_region.grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(api_frame, text="API-ключ:", style="CardMuted.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))
        key_row = ttk.Frame(api_frame, style="Card.TFrame")
        key_row.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Entry(key_row, textvariable=v_ai_key_new, width=30, show="•").pack(side="left")
        ttk.Button(key_row, text="Сохранить ключ", command=lambda: on_save_key()).pack(side="left", padx=(6, 0))
        ttk.Button(key_row, text="Удалить", command=lambda: on_delete_key()).pack(side="left", padx=(6, 0))
        lbl_key_state = ttk.Label(api_frame, style="CardMuted.TLabel")
        lbl_key_state.grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(2, 0))
        ttk.Label(api_frame, text="Модель:", style="CardMuted.TLabel").grid(row=3, column=0, sticky="w", pady=(6, 0))
        model_row = ttk.Frame(api_frame, style="Card.TFrame")
        model_row.grid(row=3, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        cb_kmodel = ttk.Combobox(model_row, textvariable=v_ai_model, values=KIMI_MODELS, width=30)
        cb_kmodel.pack(side="left")
        state["cb_kmodel"] = cb_kmodel
        ttk.Button(model_row, text="Обновить каталог", command=lambda: on_refresh_models()).pack(side="left", padx=(6, 0))

        def refresh_key_state():
            lbl_key_state.config(text="ключ сохранён (в общих настройках не хранится)" if load_key()
                                 else "ключ не задан — вставьте и нажмите «Сохранить ключ»")

        def on_save_key():
            k = v_ai_key_new.get().strip()
            if not k:
                status("Введите ключ из Kimi Code Console")
                return
            try:
                save_key(k)
                v_ai_key_new.set("")
                refresh_key_state()
                status("Ключ Kimi Code сохранён (доступ к файлу только у вас)")
            except Exception as e:
                log("сохранение ключа:", e)
                status("Не удалось сохранить ключ: " + str(e)[:80])

        def on_delete_key():
            delete_key()
            v_ai_key_new.set("")
            refresh_key_state()
            status("Ключ удалён")

        def on_refresh_models():
            if not load_key():
                status("Сначала сохраните API-ключ")
                return
            status("Запрашиваю каталог моделей…")
            region = v_ai_region.get()
            def job():
                try:
                    import kimi_code_api as kca
                    models = kca.discover_models(region, load_key())
                    q.put(("kimi_models", models, None))
                except ValueError as e:
                    q.put(("kimi_models", None, KIMI_ERRORS.get(str(e), str(e))))
                except Exception as e:
                    q.put(("kimi_models", None, str(e)[:120]))
            threading.Thread(target=job, daemon=True).start()

        refresh_key_state()
        for v in (v_ai_region, v_ai_model):
            v.trace_add("write", save_cfg)

        cmd_frame = ttk.Frame(frame, style="Card.TFrame")
        ttk.Label(cmd_frame, text="Команда (текст уходит в stdin):", style="CardMuted.TLabel").pack(anchor="w")
        ttk.Entry(cmd_frame, textvariable=v_ai_cmd, width=64).pack(anchor="w")
        v_ai_cmd.trace_add("write", save_cfg)

        def on_mode_change(*_):
            m = cb_mode.get()
            if m == "Kimi API (ключ)":
                v_ai_mode.set("api")
                cmd_frame.pack_forget()
                api_frame.pack(fill="x", pady=(8, 0))
            else:
                v_ai_mode.set("cli")
                api_frame.pack_forget()
                cmd_frame.pack(fill="x", pady=(8, 0))
                v_ai_cmd.set(AI_PRESETS.get(m, v_ai_cmd.get()))
            save_cfg()

        cb_mode.bind("<<ComboboxSelected>>", on_mode_change)
        on_mode_change()

        ttk.Label(frame, style="CardMuted.TLabel",
                  text="ответ сохранится в файл <имя>.ai.txt и переедет в папку проекта вместе с остальными"
                  ).pack(anchor="w", pady=(8, 4))

        ttk.Label(frame, text="Промпт:", style="CardMuted.TLabel").pack(anchor="w")
        txt_prompt = tk.Text(frame, height=3, wrap="word", bg=C["panel2"], fg=C["fg"],
                             insertbackground=C["fg"], relief="flat", font=("Segoe UI", 10))
        txt_prompt.pack(fill="x", pady=(2, 8))
        txt_prompt.insert("1.0", ai_prompt_text["v"])

        def close_settings():
            ai_prompt_text["v"] = txt_prompt.get("1.0", "end").strip()
            whisper_prompt_text["v"] = txt_whisper.get("1.0", "end").strip()
            save_cfg()
            win.destroy()

        ttk.Button(frame, text="Готово", style="Accent.TButton", command=close_settings).pack(anchor="e")
        win.protocol("WM_DELETE_WINDOW", close_settings)
        center_on_root(win, 660, 640)
        win.focus_set()

    # ----- шапка -----
    head = ttk.Frame(root, padding=(14, 12, 14, 4))
    head.pack(fill="x")
    ttk.Label(head, text="🎙 Транскрибатор", style="Title.TLabel").pack(side="left")
    menubtn = ttk.Menubutton(head, text="☰")
    mnu = tk.Menu(menubtn, tearoff=0, bg=C["panel2"], fg=C["fg"],
                  activebackground=C["accent"], activeforeground="#FFFFFF",
                  bd=0, font=("Segoe UI", 10))
    mnu.add_command(label="⚙  Настройки", command=lambda: open_settings())
    mnu.add_separator()
    mnu.add_command(label="Выход", command=lambda: on_close())
    menubtn.config(menu=mnu)
    menubtn.pack(side="right")
    lbl_engine = ttk.Label(head, text="", style="Muted.TLabel")
    lbl_engine.pack(side="right", padx=14)

    # ----- настройки -----
    card1 = ttk.Frame(root, style="Card.TFrame", padding=12)
    card1.pack(fill="x", padx=14, pady=(6, 6))
    ttk.Label(card1, text="Модель", style="CardMuted.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(card1, text="Устройство", style="CardMuted.TLabel").grid(row=0, column=1, sticky="w", padx=(10, 0))
    ttk.Label(card1, text="Язык", style="CardMuted.TLabel").grid(row=0, column=2, sticky="w", padx=(10, 0))
    cb_model = ttk.Combobox(card1, textvariable=v_model, values=MODEL_LIST, width=18, state="readonly")
    cb_model.grid(row=1, column=0, sticky="w", pady=(2, 0))
    cb_dev = ttk.Combobox(card1, textvariable=v_dev, values=list(DEV_MAP), width=13, state="readonly")
    cb_dev.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(2, 0))
    cb_lang = ttk.Combobox(card1, textvariable=v_lang, values=list(LANG_MAP), width=11, state="readonly")
    cb_lang.grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(2, 0))
    ttk.Checkbutton(card1, text="субтитры .srt", variable=v_srt,
                    style="Card.TCheckbutton", command=save_cfg).grid(row=1, column=3, sticky="w", padx=(16, 0), pady=(2, 0))
    ttk.Checkbutton(card1, text=".txt с таймкодами (структурированный)", variable=v_struct,
                    style="Card.TCheckbutton", command=save_cfg).grid(row=1, column=4, sticky="w", padx=(16, 0), pady=(2, 0))
    cb_dev.bind("<<ComboboxSelected>>", save_cfg)
    cb_lang.bind("<<ComboboxSelected>>", save_cfg)

    # ----- добавление файлов -----
    card2 = ttk.Frame(root, style="Card.TFrame", padding=12)
    card2.pack(fill="x", padx=14, pady=(0, 6))

    def add_files(paths):
        added = 0
        for p in paths:
            p = Path(p)
            if p.suffix.lower() not in MEDIA_EXT or not p.exists():
                continue
            key = str(p)
            if tree.exists(key):
                continue
            dur = probe_duration(p)
            est = estimate(v_model.get(), resolved_dev(), dur)
            tree.insert("", "end", iid=key, values=(
                p.name, fmt_dur(dur), "≈ " + fmt_time(est) if est else "?", "ожидает", ""),
                tags=("wait",))
            added += 1
        if added:
            update_total()

    def pick_files():
        paths = filedialog.askopenfilenames(
            title="Выберите аудио/видео файлы",
            filetypes=[("Аудио и видео", " ".join("*" + e for e in sorted(MEDIA_EXT))), ("Все файлы", "*.*")])
        if paths:
            add_files(paths)

    def pick_folder_files():
        d = filedialog.askdirectory(title="Папка с файлами для обработки")
        if d:
            add_files(sorted(Path(d).iterdir()))

    ttk.Button(card2, text="📄 Выбрать файлы…", command=pick_files).grid(row=0, column=0, sticky="w")
    ttk.Button(card2, text="📂 Обработать папку…", command=pick_folder_files).grid(row=0, column=1, sticky="w", padx=8)

    # ----- слежение и папка проекта -----
    ttk.Label(card2, text="Слежение", style="CardMuted.TLabel").grid(row=1, column=0, sticky="w", pady=(10, 0))
    row_watch = ttk.Frame(card2, style="Card.TFrame")
    row_watch.grid(row=2, column=0, columnspan=5, sticky="ew", pady=(2, 0))
    card2.columnconfigure(4, weight=1)

    ent_watch = ttk.Entry(row_watch, textvariable=v_watch, width=56)
    ent_watch.pack(side="left")

    def pick_watch():
        d = filedialog.askdirectory(title="Папка для слежения")
        if d:
            v_watch.set(d)
            save_cfg()

    def toggle_watch():
        if state["watching"]:
            state["watching"] = False
            btn_watch.config(text="▶ Начать слежение")
            status("Слежение остановлено")
            return
        d = Path(v_watch.get().strip())
        if not d.is_dir():
            messagebox.showwarning("Транскрибатор", "Выберите существующую папку для слежения.")
            return
        state["watch_dir"] = str(d)
        state["seen"] = {str(p) for p in d.iterdir() if p.is_file()}
        state["watching"] = True
        btn_watch.config(text="⏹ Остановить")
        status(f"Слежу за {d} — новые файлы попадут в очередь")
        save_cfg()
        threading.Thread(target=watch_loop, args=(d,), daemon=True).start()

    def file_ready(p: Path, prev_size) -> bool:
        """Файл готов: размер не менялся между проверками и он не занят другим процессом."""
        try:
            size = p.stat().st_size
        except OSError:
            return False
        if prev_size is None or size != prev_size or size == 0:
            return False
        try:
            os.rename(str(p), str(p))  # на Windows упадёт, если файл открыт без share (OBS и т.п.)
        except OSError:
            return False
        try:
            with open(p, "ab"):  # дополнительная проверка доступа на запись
                pass
        except OSError:
            return False
        return True

    def watch_loop(d: Path):
        stable = {}  # путь -> (размер, сколько проверок подряд размер не менялся)
        while state["watching"]:
            try:
                seen_now = set()
                for p in d.iterdir():
                    key = str(p)
                    if not p.is_file() or p.suffix.lower() not in MEDIA_EXT or key in state["seen"]:
                        continue
                    seen_now.add(key)
                    try:
                        size = p.stat().st_size
                    except OSError:
                        continue
                    prev = stable.get(key)
                    if prev and file_ready(p, prev[0]) and prev[1] >= 2:
                        # размер стабилен 3 проверки подряд (~6 сек) и файл не занят
                        stable.pop(key, None)
                        state["seen"].add(key)
                        q.put(("add", p))
                    elif prev and prev[0] == size:
                        stable[key] = (size, prev[1] + 1)
                    else:
                        stable[key] = (size, 1)
                # убрать из ожидания исчезнувшие файлы
                for key in [k for k in stable if k not in seen_now]:
                    stable.pop(key, None)
            except Exception as e:
                log("watch:", e)
            time.sleep(2)

    ttk.Button(row_watch, text="…", width=3, command=pick_watch).pack(side="left", padx=(4, 8))
    btn_watch = ttk.Button(row_watch, text="▶ Начать слежение", command=toggle_watch)
    btn_watch.pack(side="left")
    ttk.Checkbutton(row_watch, text="обрабатывать сразу", variable=v_auto,
                    style="Card.TCheckbutton", command=save_cfg).pack(side="left", padx=10)

    ttk.Label(card2, text="Папка проекта", style="CardMuted.TLabel").grid(row=3, column=0, sticky="w", pady=(10, 0))
    row_proj = ttk.Frame(card2, style="Card.TFrame")
    row_proj.grid(row=4, column=0, columnspan=5, sticky="ew", pady=(2, 0))
    ent_proj = ttk.Entry(row_proj, textvariable=v_project, width=56)
    ent_proj.pack(side="left")

    def pick_project():
        d = filedialog.askdirectory(title="Куда переносить файл и результаты после обработки")
        if d:
            v_project.set(d)
            save_cfg()

    def clear_project():
        v_project.set("")
        save_cfg()

    ttk.Button(row_proj, text="…", width=3, command=pick_project).pack(side="left", padx=(4, 8))
    ttk.Button(row_proj, text="✕ не переносить", command=clear_project).pack(side="left")
    ttk.Label(row_proj, style="CardMuted.TLabel",
              text="после обработки исходник и .txt/.srt переедут сюда; пусто — всё останется рядом с файлом"
              ).pack(side="left", padx=10)

    # ----- нижняя панель (пакуем первой, чтобы не обрезалась) -----
    bottom = ttk.Frame(root, padding=(14, 4, 14, 12))
    bottom.pack(side="bottom", fill="x")
    btns = ttk.Frame(bottom)
    btns.pack(fill="x")

    def status(t):
        lbl_status.config(text=t)
        root.update_idletasks()

    def clear_done():
        for iid in list(tree.get_children()):
            if tree.item(iid, "values")[3].startswith("готово"):
                tree.delete(iid)
        update_total()

    def open_result():
        sel = tree.selection()
        if not sel:
            messagebox.showinfo("Транскрибатор", "Выберите обработанный файл в списке.")
            return
        txt = state["results"].get(sel[0])
        if txt and Path(txt).exists():
            os.startfile(txt)
        else:
            messagebox.showinfo("Транскрибатор", "Для этого файла результат ещё не готов.")

    btn_start = ttk.Button(btns, text="▶ Начать обработку", style="Accent.TButton",
                           command=lambda: start_processing(auto=False))
    btn_start.pack(side="left")
    ttk.Button(btns, text="⏹ Стоп после текущего",
               command=lambda: state.update(stop=True)).pack(side="left", padx=8)
    ttk.Button(btns, text="🗑 Очистить готовые", command=clear_done).pack(side="left")
    ttk.Button(btns, text="📄 Открыть результат", command=open_result).pack(side="left", padx=8)

    bar = ttk.Progressbar(bottom, mode="determinate", style="accent.Horizontal.TProgressbar")
    bar.pack(fill="x", pady=(10, 4))
    lbl_status = ttk.Label(bottom, text="Готов к работе", style="Muted.TLabel")
    lbl_status.pack(fill="x")
    lbl_total = ttk.Label(bottom, text="", style="Muted.TLabel")
    lbl_total.pack(fill="x", pady=(2, 0))

    # ----- очередь -----
    qcard = ttk.Frame(root, style="Card.TFrame", padding=6)
    qcard.pack(fill="both", expand=True, padx=14, pady=(0, 6))
    cols = ("file", "dur", "est", "status", "real")
    tree = ttk.Treeview(qcard, columns=cols, show="headings", height=10)
    tree.heading("file", text="Файл")
    tree.heading("dur", text="Длительность")
    tree.heading("est", text="Оценка времени")
    tree.heading("status", text="Статус")
    tree.heading("real", text="Фактически")
    tree.column("file", width=400, anchor="w")
    tree.column("dur", width=100, anchor="center")
    tree.column("est", width=120, anchor="center")
    tree.column("status", width=230, anchor="center")
    tree.column("real", width=110, anchor="center")
    tree.tag_configure("done", foreground=C["green"])
    tree.tag_configure("busy", foreground=C["accent2"])
    tree.tag_configure("error", foreground=C["red"])
    tree.tag_configure("wait", foreground=C["muted"])
    tree.pack(fill="both", expand=True)

    def on_double(event):
        sel = tree.selection()
        if sel and sel[0] in state["results"]:
            os.startfile(state["results"][sel[0]])

    tree.bind("<Double-1>", on_double)

    def row_tag(status_text: str) -> str:
        if status_text.startswith("готово"):
            return "done"
        if status_text.startswith("ошибка"):
            return "error"
        if status_text == "обрабатываю…":
            return "busy"
        return "wait"

    def refresh_estimates(*_):
        dev = resolved_dev()
        for iid in tree.get_children():
            vals = list(tree.item(iid, "values"))
            if vals[3] == "ожидает":
                dur = probe_duration(Path(iid))
                est = estimate(v_model.get(), dev, dur)
                vals[2] = "≈ " + fmt_time(est) if est else "?"
                tree.item(iid, values=vals)
        update_total()
        save_cfg()

    cb_model.bind("<<ComboboxSelected>>", refresh_estimates)

    def update_total():
        total = 0.0
        have = False
        for iid in tree.get_children():
            vals = tree.item(iid, "values")
            if vals[3] == "ожидает":
                dur = probe_duration(Path(iid))
                est = estimate(v_model.get(), resolved_dev(), dur)
                if est:
                    total += est
                    have = True
        lbl_total.config(text=("В очереди есть файлы · ориентировочное время: ≈ " + fmt_time(total)) if have
                         else "Очередь пуста — добавьте файлы или включите слежение за папкой")

    # ----- обработка -----
    def start_processing(auto=False):
        if state["busy"]:
            return
        pending = [iid for iid in tree.get_children() if tree.item(iid, "values")[3] == "ожидает"]
        if not pending:
            return
        # фиксируем настройки до запуска потока (tkinter-переменные из потока читать нельзя)
        opts = {
            "model": v_model.get(), "device": DEV_MAP[v_dev.get()],
            "lang": LANG_MAP[v_lang.get()], "srt": v_srt.get(),
            "structured": v_struct.get(), "project_dir": v_project.get().strip(),
            "ai_on": v_ai_on.get(), "ai_cmd": v_ai_cmd.get(), "ai_prompt": ai_prompt_text["v"],
            "ai_mode": v_ai_mode.get(), "ai_region": v_ai_region.get(),
            "ai_model": v_ai_model.get().strip(),
            "whisper_prompt": whisper_prompt_text["v"],
        }
        state["busy"] = True
        state["stop"] = False
        state["auto_batch"] = auto
        state["auto_items"] = []
        btn_start.config(state="disabled")
        save_cfg()
        threading.Thread(target=worker, args=(pending, opts), daemon=True).start()

    def worker(pending, opts):
        try:
            dev = engine.ensure(opts["model"], opts["device"],
                                status=lambda t: q.put(("status", t)))
            q.put(("engine", dev))
            lang = opts["lang"]
            proj = opts["project_dir"]
            for iid in pending:
                if state["stop"]:
                    q.put(("row", iid, "пропущен"))
                    continue
                p = Path(iid)
                q.put(("row", iid, "обрабатываю…"))
                q.put(("status", f"Распознаю: {p.name}"))
                t0 = time.time()
                try:
                    dur = probe_duration(p)
                    segs = engine.transcribe(p, lang, progress=lambda f: q.put(("progress", f)),
                                             prompt=opts["whisper_prompt"])
                    dt = time.time() - t0
                    txt_path = p.with_suffix(".txt")
                    plain_text = " ".join(s[2] for s in segs).strip()
                    if opts["structured"]:
                        txt_path.write_text(
                            "\n".join(f"[{fmt_dur(a)} – {fmt_dur(b)}] {t}" for a, b, t in segs) + "\n",
                            encoding="utf-8")
                    else:
                        txt_path.write_text(plain_text + "\n", encoding="utf-8")
                    srt_path = None
                    if opts["srt"]:
                        srt_path = p.with_suffix(".srt")
                        srt_path.write_text("\n".join(
                            f"{i}\n{srt_ts(a)} --> {srt_ts(b)}\n{t}\n"
                            for i, (a, b, t) in enumerate(segs, 1)), encoding="utf-8")
                    ai_path = None
                    if opts["ai_on"] and plain_text:
                        if opts["ai_mode"] == "api" and load_key():
                            q.put(("status", f"ИИ-обработка ({opts['ai_model']}): {p.name}"))
                            try:
                                ai_out = run_kimi_text(opts["ai_region"], opts["ai_model"],
                                                       opts["ai_prompt"], plain_text)
                            except RuntimeError as e:
                                ai_out = None
                                q.put(("status", "Kimi API: " + str(e)))
                        elif opts["ai_mode"] != "api" and opts["ai_cmd"].strip():
                            q.put(("status", f"ИИ-обработка: {p.name}"))
                            ai_out = run_ai(opts["ai_cmd"], opts["ai_prompt"], plain_text)
                        else:
                            ai_out = None
                            q.put(("status", "ИИ-обработка включена, но не задан ключ/команда — пропускаю"))
                        if ai_out:
                            ai_path = p.with_suffix(".ai.txt")
                            ai_path.write_text(ai_out + "\n", encoding="utf-8")
                    moved_note = ""
                    artifacts = {"src": p, "txt": txt_path}
                    if srt_path:
                        artifacts["srt"] = srt_path
                    if ai_path:
                        artifacts["ai"] = ai_path
                    if proj:
                        moved = move_to_project(list(artifacts.values()), proj)
                        artifacts = {k: Path(moved.get(str(v), str(v))) for k, v in artifacts.items()}
                        txt_path = artifacts["txt"]
                        moved_note = " → проект"
                        log("перенесено в проект:", moved)
                    # уточняем калибровку скорости
                    if dur and dt > 0:
                        key = f"{opts['model']}|{dev}"
                        f_new = dur / max(dt - OVERHEAD_S, 0.5)
                        old = state["measured"].get(key)
                        state["measured"][key] = round(f_new if old is None else (old + f_new) / 2, 2)
                        save_cfg()
                    q.put(("done", iid, dt, str(txt_path), moved_note,
                           {k: str(v) for k, v in artifacts.items()}))
                except Exception as e:
                    log(traceback.format_exc())
                    q.put(("row", iid, "ошибка: " + str(e)[:60]))
            q.put(("status", "Обработка завершена"))
        except Exception as e:
            log(traceback.format_exc())
            q.put(("status", "Ошибка: " + str(e)))
        finally:
            q.put(("idle",))

    # ----- трей -----
    tray_icon = {"obj": None}

    def start_tray():
        try:
            import pystray
            from PIL import Image, ImageDraw
            # зелёный круг с документом — отличается от синего микрофона «Диктовки»
            img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.ellipse((2, 2, 62, 62), fill=C["green"])
            d.rounded_rectangle((20, 12, 44, 52), 4, fill="white")
            for y in (23, 32, 41):
                d.line((26, y, 38, y), fill=C["green"], width=3)

            def show(icon, item):
                q.put(("toggle_window",))

            def quit_(icon, item):
                icon.stop()
                q.put(("quit",))

            icon = pystray.Icon("transcriber", img, "Транскрибатор", menu=pystray.Menu(
                pystray.MenuItem("Показать / скрыть", show, default=True),
                pystray.MenuItem("Выход", quit_)))
            tray_icon["obj"] = icon
            icon.run_detached()
        except Exception as e:
            log("трей:", e)

    def toggle_window():
        if root.state() == "withdrawn":
            root.deiconify()
            root.lift()
        else:
            root.withdraw()

    def center_on_root(win, w, h):
        """Расположить окно по центру главного окна программы."""
        root.update_idletasks()
        x = root.winfo_x() + max((root.winfo_width() - w) // 2, 0)
        y = root.winfo_y() + max((root.winfo_height() - h) // 2, 0)
        win.geometry(f"{w}x{h}+{max(x, 0)}+{max(y, 0)}")

    def show_done_dialog(items):
        """Всплывающее окно после автообработки: куда деть готовые файлы."""
        win = tk.Toplevel(root)
        win.title("Транскрибация завершена")
        win.configure(bg=C["bg"])
        win.transient(root)
        frame = ttk.Frame(win, style="Card.TFrame", padding=14)
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        ttk.Label(frame, text=f"✅ Обработано файлов: {len(items)}",
                  style="Card.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        lst = tk.Listbox(frame, height=5, bg=C["panel2"], fg=C["fg"], relief="flat",
                         highlightthickness=0, font=("Segoe UI", 10), activestyle="none")
        for it in items:
            lst.insert("end", "• " + it["name"])
        lst.pack(fill="x", pady=(8, 4))
        lbl_loc = ttk.Label(frame, style="CardMuted.TLabel", wraplength=500)
        lbl_loc.pack(anchor="w", pady=(0, 10))

        def refresh_loc():
            lbl_loc.config(text="Сейчас файлы лежат: " +
                           str(Path(next(iter(items[0]["files"].values()))).parent))

        refresh_loc()

        def move_all():
            d = filedialog.askdirectory(parent=win, title="Куда переместить готовые файлы")
            if not d:
                return
            for it in items:
                moved = move_to_project(list(it["files"].values()), d)
                it["files"] = {k: moved.get(v, v) for k, v in it["files"].items()}
                if it["iid"] in state["results"] and "txt" in it["files"]:
                    state["results"][it["iid"]] = it["files"]["txt"]
            refresh_loc()
            status("Файлы перенесены в " + d)

        def open_loc():
            os.startfile(str(Path(next(iter(items[0]["files"].values()))).parent))

        rowb = ttk.Frame(frame, style="Card.TFrame")
        rowb.pack(fill="x", pady=(4, 0))
        ttk.Button(rowb, text="📂 Переместить в папку…", style="Accent.TButton",
                   command=move_all).pack(side="left")
        ttk.Button(rowb, text="Открыть расположение", command=open_loc).pack(side="left", padx=8)
        ttk.Button(rowb, text="Оставить как есть", command=win.destroy).pack(side="right")
        center_on_root(win, 560, 340)
        win.focus_set()

    # ----- события из потоков -----
    def poll():
        try:
            while True:
                ev = q.get_nowait()
                kind = ev[0]
                if kind == "add":
                    add_files([ev[1]])
                    if v_auto.get():
                        start_processing(auto=True)
                elif kind == "row":
                    vals = list(tree.item(ev[1], "values"))
                    vals[3] = ev[2]
                    tree.item(ev[1], values=vals, tags=(row_tag(ev[2]),))
                elif kind == "done":
                    _, iid, dt, txt, note, files = ev
                    vals = list(tree.item(iid, "values"))
                    vals[3] = "готово → " + Path(txt).name + note
                    vals[4] = fmt_time(dt)
                    tree.item(iid, values=vals, tags=("done",))
                    state["results"][iid] = txt
                    if state["auto_batch"]:
                        state["auto_items"].append({"iid": iid, "name": Path(iid).name, "files": files})
                    status(f"Готово: {Path(iid).name} за {fmt_time(dt)} — двойной клик открывает текст")
                    bar["value"] = 0
                elif kind == "progress":
                    bar["value"] = ev[1] * 100
                elif kind == "status":
                    status(ev[1])
                elif kind == "kimi_models":
                    _, models, err = ev
                    if models:
                        cb = state.get("cb_kmodel")
                        if cb is not None and cb.winfo_exists():
                            cb.config(values=models)
                        if v_ai_model.get() not in models:
                            v_ai_model.set(models[0])
                        save_cfg()
                        status("Каталог моделей обновлён: " + ", ".join(models))
                    else:
                        status("Не удалось получить каталог: " + (err or "?"))
                elif kind == "engine":
                    lbl_engine.config(text="движок: видеокарта" if ev[1] == "cuda" else "движок: процессор")
                elif kind == "toggle_window":
                    toggle_window()
                elif kind == "quit":
                    on_close()
                    return
                elif kind == "idle":
                    state["busy"] = False
                    btn_start.config(state="normal")
                    update_total()
                    # всплывающее окно после автоматической обработки из отслеживаемой папки
                    if state["auto_batch"] and state["auto_items"]:
                        items = state["auto_items"]
                        state["auto_items"] = []
                        if root.state() == "withdrawn":
                            root.deiconify()
                        show_done_dialog(items)
                    state["auto_batch"] = False
        except queue.Empty:
            pass
        root.after(100, poll)

    def on_close():
        state["watching"] = False
        save_cfg()
        try:
            if tray_icon["obj"]:
                tray_icon["obj"].stop()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    start_tray()
    update_total()
    poll()
    root.mainloop()

if __name__ == "__main__":
    enable_dpi_awareness()
    if "--selftest" in sys.argv:
        selftest()
    else:
        try:
            run_gui()
        except Exception:
            log(traceback.format_exc())
            raise
