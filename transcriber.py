# Транскрибатор – быстрый перевод аудио/видео в текст (faster-whisper, GPU).
# Выбор модели, файлы вручную или слежение за папкой, оценка времени,
# перенос исходника и результатов в папку проекта, трей, автозапуск.

import os, sys, json, time, threading, queue, shutil, ctypes, subprocess, traceback, uuid
import urllib.request, urllib.error
from pathlib import Path

IS_WIN = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"

# ---------- пути и окружение ----------

def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        if IS_MAC:
            # .app-бандл нельзя редактировать — конфиг и модели в Application Support
            d = Path.home() / "Library/Application Support/Transcriber"
            d.mkdir(parents=True, exist_ok=True)
            return d
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
GO_KEY_PATH = PRIVATE_DIR / "go_key"

def open_file(path):
    """Открыть файл штатным приложением ОС."""
    if IS_WIN:
        os.startfile(str(path))
    elif IS_MAC:
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])

# ----- защищённое хранение ключа Kimi Code (отдельно от конфига) -----

def save_key(key: str):
    """Атомарно записать ключ; доступ только у текущего пользователя."""
    PRIVATE_DIR.mkdir(exist_ok=True)
    tmp = KEY_PATH.with_suffix(".tmp")
    tmp.write_text(key.strip(), encoding="utf-8")
    os.replace(tmp, KEY_PATH)
    try:  # доступ к ключу только у текущего пользователя
        if IS_WIN:
            subprocess.run(["icacls", str(KEY_PATH), "/inheritance:r", "/grant:r",
                            f"{os.environ.get('USERDOMAIN', '')}\\{os.environ.get('USERNAME', '')}:F"],
                           capture_output=True, creationflags=0x08000000)  # CREATE_NO_WINDOW
        else:
            os.chmod(KEY_PATH, 0o600)
    except Exception as e:
        log("права на ключ:", e)

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

MEDIA_EXT = {
    # аудио
    ".mp3", ".mp2", ".wav", ".m4a", ".m4b", ".aac", ".flac", ".ogg", ".opus",
    ".wma", ".aiff", ".aif", ".caf", ".amr", ".ac3", ".mka", ".wv", ".ape",
    # видео
    ".mp4", ".m4v", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".asf", ".flv",
    ".f4v", ".mpg", ".mpeg", ".mpe", ".m2v", ".ts", ".mts", ".m2ts", ".m2t",
    ".3gp", ".3g2", ".vob", ".ogv", ".rm", ".rmvb", ".divx", ".dv", ".mxf",
}

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
    """Чёткие шрифты на мониторах с масштабированием (Windows)."""
    if not IS_WIN:
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

def dark_titlebar(root):
    """Тёмный заголовок окна (Windows 10 1809+ / 11)."""
    if not IS_WIN:
        return
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

def autostart_plist() -> Path:
    return Path.home() / "Library/LaunchAgents/ai.frozen12.transcriber.plist"

def autostart_enabled() -> bool:
    if IS_MAC:
        return autostart_plist().exists()
    try:
        return autostart_link().exists()
    except Exception:
        return False

def set_autostart(enable: bool):
    """Автозапуск: ярлык в Startup (Windows) или LaunchAgents (macOS)."""
    try:
        if IS_MAC:
            pl = autostart_plist()
            if enable:
                pl.parent.mkdir(parents=True, exist_ok=True)
                pl.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>ai.frozen12.transcriber</string>
<key>ProgramArguments</key><array><string>{}</string></array>
<key>WorkingDirectory</key><string>{}</string>
<key>RunAtLoad</key><true/>
</dict></plist>
""".format(Path(sys.executable).resolve(), BASE), encoding="utf-8")
            else:
                pl.unlink(missing_ok=True)
            return
        lnk = autostart_link()
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

# ----- Unicode drag&drop для Tk (Windows) -----
# У каждого Tk-виджета свой HWND; Explorer отправляет WM_DROPFILES окну под
# курсором. Штатный windnd читает ANSI-имена (ломает кириллицу), поэтому
# регистрируем свой WndProc и читаем DragQueryFileW.

_DROP_HOOKS = []   # (hwnd, old_proc, new_proc) — держим ссылки, иначе ctypes освободит callback
_DROP_HWNDS = set()
_DROP_QUEUE = queue.Queue()   # пути из WM_DROPFILES ждут разбора в главном потоке Tk

def drain_dropped_files():
    """Забрать пути, принесённые drag&drop. Вызывать только из главного потока Tk."""
    out = []
    while True:
        try:
            out.append(_DROP_QUEUE.get_nowait())
        except queue.Empty:
            return out

def hook_dropfiles_unicode(widget):
    """Разрешить перетаскивание файлов на виджет; пути попадают в _DROP_QUEUE."""
    if not IS_WIN:
        return False
    from ctypes import wintypes
    try:
        hwnd = int(widget.winfo_id())
    except Exception:
        return False
    if hwnd in _DROP_HWNDS:
        return True
    WM_DROPFILES = 0x0233
    GWL_WNDPROC = -4
    user32 = ctypes.windll.user32
    shell32 = ctypes.windll.shell32
    WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT,
                                 wintypes.WPARAM, wintypes.LPARAM)
    user32.GetWindowLongPtrW.restype = ctypes.c_void_p
    user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.SetWindowLongPtrW.restype = ctypes.c_void_p
    user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
    user32.CallWindowProcW.restype = ctypes.c_ssize_t
    user32.CallWindowProcW.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT,
                                       wintypes.WPARAM, wintypes.LPARAM]
    shell32.DragQueryFileW.restype = wintypes.UINT
    shell32.DragQueryFileW.argtypes = [wintypes.HANDLE, wintypes.UINT,
                                       wintypes.LPWSTR, wintypes.UINT]
    shell32.DragFinish.argtypes = [wintypes.HANDLE]
    shell32.DragAcceptFiles.argtypes = [wintypes.HWND, wintypes.BOOL]

    old_proc = user32.GetWindowLongPtrW(hwnd, GWL_WNDPROC)

    def wndproc(h, msg, wp, lp):
        if msg == WM_DROPFILES:
            hdrop = wintypes.HANDLE(wp)
            try:
                count = shell32.DragQueryFileW(hdrop, 0xFFFFFFFF, None, 0)
                files = []
                for i in range(count):
                    n = shell32.DragQueryFileW(hdrop, i, None, 0)
                    if not n:
                        continue
                    buf = ctypes.create_unicode_buffer(n + 1)
                    shell32.DragQueryFileW(hdrop, i, buf, n + 1)
                    files.append(buf.value)
            except Exception as e:
                log("drag&drop чтение:", e)
                files = []
            finally:
                shell32.DragFinish(hdrop)
            if files:
                # Внутри WndProc нельзя обращаться к Tk/Tcl: Проводник присылает
                # WM_DROPFILES, пока Tk ждёт сообщения, и повторный вход в Tcl
                # роняет процесс (0xc0000409). Поэтому только кладём пути в
                # очередь — разберёт её poll() в главном потоке.
                try:
                    _DROP_QUEUE.put(files)
                except Exception as e:
                    log("drag&drop очередь:", e)
        return user32.CallWindowProcW(old_proc, h, msg, wp, lp)

    proc = WNDPROC(wndproc)
    user32.SetWindowLongPtrW(hwnd, GWL_WNDPROC, ctypes.cast(proc, ctypes.c_void_p))
    shell32.DragAcceptFiles(hwnd, True)
    _DROP_HOOKS.append((hwnd, old_proc, proc))
    _DROP_HWNDS.add(hwnd)
    return True

def hook_dropfiles_tree(widget):
    """Повесить приём файлов на виджет и всех его потомков."""
    if not IS_WIN:
        return
    def hook_all(w):
        try:
            hook_dropfiles_unicode(w)
        except Exception as e:
            log("drag&drop hook:", w, e)
        try:
            children = w.winfo_children()
        except Exception:
            children = []
        for child in children:
            hook_all(child)
    hook_all(widget)

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

GO_MODELS_URL = "https://opencode.ai/inference/go/v1/models"
GO_MODELS_DEFAULT = ["qwen3.8-flash", "deepseek-v4.1-flash", "glm-5.3-flash",
                     "kimi-k3", "kimi-k2.7-code", "minimax-m3.1-flash", "hy3"]

def save_go_key(key: str):
    """Ключ подписки OpenCode Go — в защищённое хранилище; в конфиг и логи не попадает."""
    PRIVATE_DIR.mkdir(exist_ok=True)
    tmp = GO_KEY_PATH.with_suffix(".tmp")
    tmp.write_text(key.strip(), encoding="utf-8")
    os.replace(tmp, GO_KEY_PATH)
    try:
        if IS_WIN:
            subprocess.run(["icacls", str(GO_KEY_PATH), "/inheritance:r", "/grant:r",
                            f"{os.environ.get('USERDOMAIN', '')}\\{os.environ.get('USERNAME', '')}:F"],
                           capture_output=True, creationflags=0x08000000)
        else:
            os.chmod(GO_KEY_PATH, 0o600)
    except Exception as e:
        log("права на ключ Go:", e)

def load_go_key() -> str:
    try:
        return GO_KEY_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

def delete_go_key():
    try:
        GO_KEY_PATH.unlink(missing_ok=True)
    except Exception:
        pass

def go_fetch_catalog(key: str):
    """Список моделей подписки Go с уровнями рассуждения.
    Возвращает [{"id": "...", "efforts": ["low", ...]}]. Требует собственный User-Agent (models.dev отдаёт 403 без него)."""
    import urllib.request
    import urllib.error
    hdr = {"User-Agent": "Transcriber/1.2 (+https://github.com/Frozen-12/Transcriber)"}
    if key:
        hdr["Authorization"] = "Bearer " + key
    try:
        r = urllib.request.Request(GO_MODELS_URL, headers=hdr)
        with urllib.request.urlopen(r, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RuntimeError("ключ OpenCode Go не принят каталогом (" + str(e.code) + ")")
        raise RuntimeError("каталог моделей недоступен (" + str(e.code) + ")")
    except Exception as e:
        raise RuntimeError("нет доступа к списку моделей: " + str(e)[:100])
    items = data.get("data", data) if isinstance(data, dict) else data
    out = []
    for m in items if isinstance(items, list) else []:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        efforts = []
        caps = m.get("capabilities") or {}
        rea = caps.get("reasoning") or {}
        eff = rea.get("effort")
        if isinstance(eff, list):
            efforts = [str(x) for x in eff]
        elif isinstance(eff, dict):
            vals = eff.get("values") or eff.get("levels") or []
            efforts = [str(x) for x in vals if isinstance(vals, list)]
        out.append({"id": str(m["id"]), "efforts": efforts})
    return out or [{"id": x, "efforts": []} for x in GO_MODELS_DEFAULT]

def _go_error_map(err):
    """Ответ OpenCode Go с ошибкой → человекочитаемая причина."""
    et = str(err.get("type") or "")
    status = err.get("status")
    msg = str(err.get("message") or "")
    low = (et + " " + msg).lower()
    if status == 401 or "auth" in low or "credential" in low:
        return "ключ отклонён (401) — проверьте ключ OpenCode Go в opencode.ai/console"
    if status == 402 or "payment" in low or "subscription" in low:
        return "нет доступа по подписке (402) — проверьте тариф Go в opencode.ai/console"
    if status == 403 or "region" in low:
        return "доступ запрещён (403) — модель недоступна для вашего региона/настроек приватности"
    if status == 429 or "quota" in low or "usage limit" in low or "allowance" in low:
        return "лимит тарифа исчерпан (429) — см. Usage в opencode.ai/console"
    if "unavailable" in low or "not found" in low:
        return "модель недоступна — выберите другую из списка"
    return "ошибка OpenCode Go: " + (msg or et or "неизвестно")[:160]

GO_API_URL = "https://opencode.ai/inference/go/openai/v1/chat/completions"
GO_UA = "Transcriber/1.4 (+https://github.com/Frozen-12/Transcriber)"

def _go_http_error(status: int, raw: bytes) -> RuntimeError:
    """HTTP-ошибка OpenCode Go → понятная причина."""
    data = None
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        pass
    err = {}
    if isinstance(data, dict):
        err = data.get("error") if isinstance(data.get("error"), dict) else data
        if not err.get("message") and data.get("message"):
            err["message"] = data["message"]
    msg = str((err or {}).get("message") or raw[:200].decode("utf-8", "replace"))
    return RuntimeError(_go_error_map({"type": (err or {}).get("type") or "",
                                       "status": status, "message": msg}))

def run_opencode_go(model_id: str, effort: str, prompt: str, text: str, timeout=1800):
    """Один вызов OpenCode Go напрямую по API-ключу (OpenAI-совместимый endpoint).
    CLI не требуется. Возвращает текст конспекта."""
    key = load_go_key()
    if not key:
        raise RuntimeError("ключ OpenCode Go не задан (Настройки → OpenCode Go)")
    mid = (model_id or "").strip() or "qwen3.8-flash"
    if "/" in mid:
        mid = mid.split("/", 1)[1]  # на случай opencode-go/model или provider/model
    if "#" in mid:
        mid, embedded_effort = mid.split("#", 1)
        effort = effort or embedded_effort
    body_text = (prompt + "\n\n" + text) if prompt else text
    payload = {
        "model": mid,
        "messages": [{"role": "user", "content": body_text}],
        "stream": False,
        "max_tokens": 8192,
    }
    if effort:
        payload["reasoning_effort"] = effort
    req = urllib.request.Request(
        GO_API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "User-Agent": GO_UA,
            # OpenCode Go просит стабильный идентификатор разговора для маршрутизации
            "x-opencode-session": uuid.uuid4().hex,
        })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise _go_http_error(e.code, e.read()) from None
    except TimeoutError:
        raise RuntimeError("OpenCode Go не ответил за " + str(timeout // 60) + " минут") from None
    except Exception as e:
        raise RuntimeError("нет связи с OpenCode Go: " + str(e)[:160]) from None
    try:
        reply = json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        raise RuntimeError("неожиданный ответ OpenCode Go") from None
    choices = reply.get("choices") if isinstance(reply, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("OpenCode Go вернул ответ без текста")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):  # часть провайдеров отдаёт массив частей
        content = "".join(str(c.get("text") or "") for c in content if isinstance(c, dict))
    out = content.strip() if isinstance(content, str) else ""
    if not out:
        raise RuntimeError("пустой ответ модели OpenCode Go")
    usage = reply.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    log("go usage:", usage.get("prompt_tokens"), "->", usage.get("completion_tokens"),
        "reasoning:", details.get("reasoning_tokens"), "модель:", mid)
    return out

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

def find_opencode_cli() -> str:
    """Полный путь к opencode CLI, если установлен (desktop-версия не добавляет его в PATH)."""
    cands = []
    if IS_WIN:
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        cands.append(local / r"Programs\ai.opencode.desktop\resources\opencode-cli.exe")
        cands.append(local / r"Programs\OpenCode\resources\opencode-cli.exe")
    else:
        cands.append(Path("/usr/local/bin/opencode"))
        cands.append(Path.home() / ".opencode/bin/opencode")
        cands.append(Path.home() / ".local/bin/opencode")
    for p in cands:
        try:
            if p.is_file():
                return str(p)
        except OSError:
            pass
    for name in ("opencode-cli.exe", "opencode"):
        found = shutil.which(name)
        if found:
            return found
    return ""

def run_opencode(cli_path: str, model: str, prompt: str, text: str):
    """Свежая сессия OpenCode на вызов: CLI читает stdin и печатает ответ в stdout."""
    cmd = [cli_path, "run", "--auto", "--title", "Транскрибатор", "--format", "default"]
    if model.strip():
        cmd += ["--model", model.strip()]
    try:
        proc = subprocess.run(cmd, input=(prompt + "\n\n" + text) if prompt else text,
                              capture_output=True, timeout=1800,
                              encoding="utf-8", errors="replace")
        out = (proc.stdout or "").strip()
        if proc.returncode != 0:
            log("OpenCode завершился с кодом", proc.returncode, (proc.stderr or "")[:500])
        return out or None
    except Exception as e:
        log("OpenCode-обработка:", e)
        return None

def run_opencode_server(base_url: str, key: str, model_spec: str, prompt: str, text: str):
    """Конспект через OpenCode Server API: свежая сессия, prompt, опрос ответа.
    model_spec вида provider/model[#variant]; пусто = модель по умолчанию сервера.
    Возвращает текст ответа или None."""
    import urllib.request
    import urllib.error
    base = base_url.strip().rstrip("/")
    if not base.startswith("http"):
        base = "http://" + base
    hdr = {"Content-Type": "application/json"}
    if key:
        import base64
        tok = base64.b64encode(("opencode:" + key).encode("utf-8")).decode()
        hdr["Authorization"] = "Basic " + tok

    def req(method, path, body=None, timeout=60):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        r = urllib.request.Request(base + path, data=data, method=method, headers=hdr)
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else None

    def unwrap(x):
        return x.get("data", x) if isinstance(x, dict) else x

    try:
        sid = unwrap(req("POST", "/api/session", {"title": "Транскрибатор"})).get("id")
        if not sid:
            raise RuntimeError("сервер не вернул id сессии")
        if model_spec.strip():
            spec = model_spec.strip()
            provider_id, model_id = spec.split("/", 1) if "/" in spec else ("", spec)
            model_id, variant = model_id.split("#", 1) if "#" in model_id else (model_id, "")
            ref = {"id": model_id}
            if provider_id:
                ref["providerID"] = provider_id
            if variant:
                ref["variant"] = variant
            try:
                req("POST", "/api/session/" + sid + "/model", {"model": ref})
            except Exception as e:
                log("OpenCode server: смена модели не удалась:", e)
        full_text = (prompt + "\n\n" + text) if prompt else text
        req("POST", "/api/session/" + sid + "/prompt", {"text": full_text}, timeout=120)
        t0 = time.time()
        while time.time() - t0 < 1800:
            time.sleep(3)
            try:
                items = unwrap(req("GET", "/api/session/" + sid + "/message")) or []
            except Exception:
                continue
            assistants = [m for m in items if isinstance(m, dict) and m.get("type") == "assistant"]
            if assistants:
                last = assistants[-1]
                if (last.get("time") or {}).get("completed"):
                    parts = [c.get("text", "") for c in last.get("content", [])
                             if isinstance(c, dict) and c.get("type") == "text"]
                    out = "\n\n".join(p for p in parts if p.strip()).strip()
                    req("DELETE", "/api/session/" + sid, timeout=30)  # не копить сессии
                    return out or None
        log("OpenCode server: ожидание ответа прервано (timeout)")
        try:
            req("POST", "/api/session/" + sid + "/interrupt", None, timeout=30)
            req("DELETE", "/api/session/" + sid, timeout=30)
        except Exception:
            pass
        return None
    except Exception as e:
        log("OpenCode server:", e)
        return None

# ----- защищённое хранение ключа сервера OpenCode -----
OC_KEY_PATH = PRIVATE_DIR / "oc_server_key"

def save_oc_key(key: str):
    PRIVATE_DIR.mkdir(exist_ok=True)
    tmp = OC_KEY_PATH.with_suffix(".tmp")
    tmp.write_text(key.strip(), encoding="utf-8")
    os.replace(tmp, OC_KEY_PATH)
    try:
        if not IS_WIN:
            os.chmod(OC_KEY_PATH, 0o600)
    except Exception:
        pass

def load_oc_key() -> str:
    try:
        return OC_KEY_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

def delete_oc_key():
    try:
        OC_KEY_PATH.unlink(missing_ok=True)
    except Exception:
        pass

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
        status(f"Загрузка модели {model_name} ({'видеокарта' if dev == 'cuda' else 'процессор'})… "
               "при первом запуске модель скачивается из интернета")
        add_cuda_dlls()
        from faster_whisper import WhisperModel
        self.model = None  # освободить старую
        if dev == "cuda":
            try:
                self.model = WhisperModel(model_name, device="cuda", compute_type="float16")
            except Exception as e:
                log("CUDA не завелась, перехожу на процессор:", e)
                status("Видеокарта недоступна — продолжаю на процессоре")
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
# Дизайн-система: тёмная OLED-тема (slate-navy), единые токены цвета,
# шкала шрифтов и отступов, карточки с hairline-границей, состояния фокуса.

FAM = "Segoe UI" if IS_WIN else ("Helvetica Neue" if IS_MAC else "Noto Sans")
MONO = "Consolas" if IS_WIN else ("SF Mono" if IS_MAC else "DejaVu Sans Mono")

C = {
    # поверхности
    "bg": "#0F1524",          # холст окна
    "panel": "#171F31",       # карточки
    "panel2": "#212B40",      # поля ввода, вторичные кнопки
    "panel3": "#2A3550",      # hover полей/кнопок
    # границы
    "border": "#2B3852",      # hairline 1px (структурные разделители)
    "borderStrong": "#3D4E72",
    "field": "#5A6B8C",       # рамка полей ввода: ≥3:1 на панелях (WCAG non-text)
    # текст
    "fg": "#EDF2FA",
    "muted": "#93A1BC",
    # акценты
    "accent": "#4C8DFF",      # фокус, выделение, прогресс
    "accent2": "#7FABFF",     # hover акцента
    "accentSel": "#3D6FD9",   # фон выделенной строки (контраст с белым >= 4.5)
    "green": "#2FBF62",       # главное действие «запустить»
    "greenHover": "#4ECD7C",
    "onGreen": "#07180B",     # тёмный текст на зелёной кнопке
    "red": "#EA6860",         # ошибки (текст: ≥4.5:1 на panel)
    "redFill": "#E5484D",     # сплошные акценты ошибок (если понадобятся)
    "yellow": "#DFA233",      # предупреждения
}

# шкала отступов (8px) и типографика
S = {"xs": 4, "s": 8, "m": 12, "l": 16, "xl": 24}
F = {
    "title": (FAM, 15, "bold"),
    "section": (FAM, 11, "bold"),
    "body": (FAM, 10),
    "small": (FAM, 9),
    "kicker": (FAM, 9, "bold"),
    "mono": (MONO, 10),
    "btnBold": (FAM, 10, "bold"),
}

def apply_theme(root):
    root.configure(bg=C["bg"])
    root.option_add("*TCombobox*Listbox.background", C["panel2"])
    root.option_add("*TCombobox*Listbox.foreground", C["fg"])
    root.option_add("*TCombobox*Listbox.selectBackground", C["accentSel"])
    root.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")
    root.option_add("*Menu*background", C["panel2"])
    root.option_add("*Menu*foreground", C["fg"])
    from tkinter import ttk
    st = ttk.Style(root)
    st.theme_use("clam")

    # база: единый шрифт, цвета и hairline-границы вместо 3D-скосов clam
    st.configure(".", background=C["bg"], foreground=C["fg"], font=F["body"],
                 bordercolor=C["border"], lightcolor=C["bg"], darkcolor=C["bg"],
                 focuscolor=C["accent"], troughcolor=C["panel2"])

    # --- фреймы-карточки: фон панели + граница 1px (структура рамкой, не тенью) ---
    st.configure("TFrame", background=C["bg"])
    st.configure("Card.TFrame", background=C["panel"], relief="solid", borderwidth=1,
                 lightcolor=C["border"], darkcolor=C["border"])
    st.configure("Inner.TFrame", background=C["panel"])

    # --- надписи ---
    st.configure("TLabel", background=C["bg"], foreground=C["fg"], font=F["body"])
    st.configure("Card.TLabel", background=C["panel"], foreground=C["fg"], font=F["body"])
    st.configure("Muted.TLabel", background=C["bg"], foreground=C["muted"])
    st.configure("CardMuted.TLabel", background=C["panel"], foreground=C["muted"])
    st.configure("Kicker.TLabel", background=C["bg"], foreground=C["muted"], font=F["kicker"])
    st.configure("CardKicker.TLabel", background=C["panel"], foreground=C["muted"], font=F["kicker"])
    st.configure("Title.TLabel", background=C["bg"], foreground=C["fg"], font=F["title"])
    st.configure("Sub.TLabel", background=C["bg"], foreground=C["muted"], font=F["small"])
    st.configure("Section.TLabel", background=C["panel"], foreground=C["fg"], font=F["section"])
    st.configure("Pill.TLabel", background=C["panel2"], foreground=C["muted"],
                 font=F["small"], padding=(10, 3))
    st.configure("Status.TLabel", background=C["bg"], foreground=C["muted"], font=F["body"])
    st.configure("Mono.TLabel", background=C["bg"], foreground=C["fg"], font=F["mono"])

    # --- кнопки: вторичные (панель + граница) и основная (зелёная, тёмный текст) ---
    st.configure("TButton", background=C["panel2"], foreground=C["fg"],
                 padding=(14, 7), borderwidth=1, relief="flat",
                 lightcolor=C["border"], darkcolor=C["border"],
                 focuscolor=C["accent"], cursor="hand2")
    st.map("TButton",
           background=[("active", C["panel3"]), ("pressed", C["border"]), ("disabled", C["panel"])],
           foreground=[("disabled", C["muted"])],
           bordercolor=[("active", C["borderStrong"]), ("focus", C["accent"])])

    st.configure("Accent.TButton", background=C["green"], foreground=C["onGreen"],
                 font=F["btnBold"], padding=(18, 8), borderwidth=1,
                 lightcolor=C["green"], darkcolor=C["green"], focuscolor=C["fg"])
    st.map("Accent.TButton",
           background=[("active", C["greenHover"]), ("pressed", "#28A554"), ("disabled", C["panel"])],
           foreground=[("disabled", C["muted"])])

    st.configure("Ghost.TButton", background=C["bg"], foreground=C["muted"],
                 borderwidth=0, lightcolor=C["bg"], darkcolor=C["bg"])
    st.map("Ghost.TButton",
           background=[("active", C["panel2"]), ("pressed", C["panel3"])],
           foreground=[("active", C["fg"])])

    # вторичная кнопка-«напарник» основной: тот же полужирный шрифт → одинаковая
    # ширина при одинаковом width (единицы ширины ttk зависят от шрифта)
    st.configure("Peer.TButton", font=F["btnBold"], padding=(18, 8))

    # --- чекбоксы: индикатор подсвечивается при отметке ---
    for style_, bgv in (("TCheckbutton", C["bg"]), ("Card.TCheckbutton", C["panel"]),
                        ("TRadiobutton", C["bg"]), ("Card.TRadiobutton", C["panel"])):
        st.configure(style_, background=bgv, foreground=C["fg"], font=F["body"],
                     indicatorcolor=C["panel2"], focuscolor=C["accent"], cursor="hand2")
        st.map(style_, background=[("active", bgv)],
               indicatorcolor=[("selected", C["accent"]), ("active", C["panel3"])])

    # --- поля ввода: рамка не ниже 3:1, при фокусе — акцент ---
    st.configure("TEntry", fieldbackground=C["panel2"], foreground=C["fg"],
                 insertcolor=C["fg"], bordercolor=C["field"], padding=(6, 6),
                 lightcolor=C["field"], darkcolor=C["field"])
    st.map("TEntry",
           bordercolor=[("focus", C["accent"]), ("disabled", C["panel"])],
           lightcolor=[("focus", C["accent"]), ("disabled", C["panel"])],
           darkcolor=[("focus", C["accent"]), ("disabled", C["panel"])],
           foreground=[("disabled", C["muted"])])

    st.configure("TCombobox", fieldbackground=C["panel2"], background=C["panel2"],
                 foreground=C["fg"], arrowcolor=C["fg"], bordercolor=C["field"],
                 padding=(6, 5), lightcolor=C["field"], darkcolor=C["field"],
                 arrowsize=14, cursor="hand2")
    st.map("TCombobox",
           bordercolor=[("focus", C["accent"])],
           lightcolor=[("focus", C["accent"])], darkcolor=[("focus", C["accent"])],
           fieldbackground=[("readonly", C["panel2"]), ("disabled", C["panel"])],
           foreground=[("readonly", C["fg"]), ("disabled", C["muted"])],
           selectbackground=[("readonly", C["panel2"])],
           selectforeground=[("readonly", C["fg"])],
           arrowcolor=[("pressed", C["fg"]), ("hover", C["fg"])])

    # --- разделители и прокрутка в тёмной теме ---
    st.configure("TSeparator", background=C["border"], relief="flat")
    for sb in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
        st.configure(sb, background=C["panel3"], troughcolor=C["bg"],
                     bordercolor=C["bg"], arrowcolor=C["muted"],
                     relief="flat", borderwidth=0, arrowsize=12)
        st.map(sb, background=[("active", C["borderStrong"]), ("pressed", C["accent"])],
               arrowcolor=[("active", C["fg"])])

    # --- очередь (дерево) ---
    st.configure("Treeview", background=C["panel"], fieldbackground=C["panel"],
                 foreground=C["fg"], rowheight=30, borderwidth=0,
                 lightcolor=C["panel"], darkcolor=C["panel"], font=F["body"])
    st.configure("Treeview.Heading", background=C["panel"], foreground=C["muted"],
                 font=F["kicker"], relief="flat", borderwidth=0, padding=(6, 6),
                 lightcolor=C["border"], darkcolor=C["border"])
    st.map("Treeview",
           background=[("selected", C["accentSel"])],
           foreground=[("selected", "#FFFFFF")])
    st.map("Treeview.Heading", background=[("active", C["panel2"])])

    # --- прогресс ---
    st.configure("accent.Horizontal.TProgressbar", troughcolor=C["panel2"],
                 background=C["accent"], bordercolor=C["panel2"],
                 lightcolor=C["accent"], darkcolor=C["accent"],
                 borderwidth=0, thickness=8)

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
    root.minsize(1300, 640)

    def _restore_geom(key, default_w, default_h):
        g = str(cfg.get(key) or "")
        try:
            w, h = g.lower().split("x", 1)
            w, h = int(w), int(h)
            if w >= root.winfo_screenwidth() or h >= root.winfo_screenheight():
                raise ValueError
            root.geometry(f"{max(w, 1300)}x{max(h, 640)}")
        except Exception:
            root.geometry(f"{default_w}x{default_h}")

    _restore_geom("window_size", 1380, 760)
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
        "window_size": cfg.get("window_size"),    # "ШхВ" главного окна (пусто = дефолт)
        "settings_size": cfg.get("settings_size"),  # "ШхВ" окна настроек
        "go_catalog": [],        # кэш каталога моделей Go: [{"id","efforts"}]
        "go_verified": False,    # отдельно: «ключ сохранён» и «модельный вызов проверен»
        "last_dir": cfg.get("last_dir", ""),  # последняя папка в диалогах выбора
    }

    v_model = tk.StringVar(value=cfg.get("model", "large-v3-turbo"))
    v_dev = tk.StringVar(value={"auto": "Авто", "cuda": "Видеокарта", "cpu": "Процессор"}.get(cfg.get("device", "auto"), "Авто"))
    v_lang = tk.StringVar(value={"ru": "Русский", "auto": "Авто", "en": "Английский"}.get(cfg.get("language", "ru"), "Русский"))
    v_srt = tk.BooleanVar(value=bool(cfg.get("srt", False)))
    v_struct = tk.BooleanVar(value=bool(cfg.get("structured", True)))
    v_auto = tk.BooleanVar(value=bool(cfg.get("auto_process", True)))
    v_autostart = tk.BooleanVar(value=autostart_enabled())
    v_minimized = tk.BooleanVar(value=bool(cfg.get("start_minimized", False)))
    v_watch = tk.StringVar(value=state["watch_dir"])
    v_project = tk.StringVar(value=cfg.get("project_dir", ""))
    v_ai_on = tk.BooleanVar(value=bool(cfg.get("ai_enabled", False)))
    v_ai_cmd = tk.StringVar(value=cfg.get("ai_command", ""))
    v_ai_mode = tk.StringVar(value=cfg.get("ai_mode", "api"))       # api | cli
    v_ai_region = tk.StringVar(value=cfg.get("ai_region", "global"))  # global | mainland-cn
    v_ai_key_new = tk.StringVar(value="")   # поле ввода ключа: всегда пустое, ключ не показывается
    v_ai_model = tk.StringVar(value=cfg.get("ai_model", "kimi-for-coding"))
    v_oc_cli = tk.StringVar(value=cfg.get("oc_cli") or find_opencode_cli())
    v_oc_model = tk.StringVar(value=cfg.get("oc_model", ""))  # пусто = модель из конфига OpenCode
    v_oc_conn = tk.StringVar(value=cfg.get("oc_conn", "cli"))  # cli | server (server — через конфиг)
    v_oc_url = tk.StringVar(value=cfg.get("oc_url", "http://127.0.0.1:4096"))  # скрытая настройка
    v_oc_variant = tk.StringVar(value=cfg.get("oc_variant", ""))  # глубина рассуждения ("" = как в модели/по умолчанию)
    v_go_model = tk.StringVar(value=cfg.get("go_model", "qwen3.8-flash"))
    v_go_key_new = tk.StringVar(value="")  # поле ввода ключа Go: всегда пустое, ключ не показывается
    v_go_effort = tk.StringVar(value=cfg.get("go_effort", ""))  # уровень рассуждения Go ('' = по умолчанию модели)

    KIMI_MODELS = ["kimi-for-coding", "kimi-for-coding-highspeed",
                   "k3.1-pro", "k3.1-pro-highspeed", "k3", "k3-256k"]

    DEFAULT_AI_PROMPT = ("Сделай структурированный конспект: основные тезисы, принятые решения, "
                         "поручения с ответственными. Отвечай по-русски. Не используй в выводе "
                         "разделители \"*\", \"|\", \"#\". Используй простые текстовые подзаголовки "
                         "с двоеточием, без ## и ###.")
    DEFAULT_WHISPER_PROMPT = ("DevOps, CI/CD, Kubernetes, K8s, Docker, Helm, Terraform, Ansible, "
                              "Jenkins, GitLab, GitHub, Git, ArgoCD, Prometheus, Grafana, Alertmanager, "
                              "Loki, ELK, Elasticsearch, Kibana, Kafka, RabbitMQ, Nginx, Traefik, HAProxy, "
                              "PostgreSQL, MySQL, Redis, MongoDB, ClickHouse, S3, MinIO, Vault, Consul, "
                              "Linux, Ubuntu, Debian, CentOS, Bash, Python, Go, YAML, JSON, API, REST, "
                              "gRPC, OAuth, SSL, TLS, DNS, VPN, Docker Hub, AWS, Azure, Google Cloud, "
                              "Yandex Cloud, Selectel, VK Cloud, 1С, Битрикс24, amoCRM, ELMA365, Kaiten, "
                              "Jira, Confluence, Notion, SLA, on-call")
    AI_PRESETS = {"Kimi CLI": "kimi --print", "Ollama": "ollama run llama3.1", "Своя команда": ""}
    ai_prompt_text = {"v": cfg.get("ai_prompt") or DEFAULT_AI_PROMPT}
    whisper_prompt_text = {"v": cfg.get("whisper_prompt") or DEFAULT_WHISPER_PROMPT}

    DEV_MAP = {"Авто": "auto", "Видеокарта": "cuda", "Процессор": "cpu"}
    LANG_MAP = {"Русский": "ru", "Авто": None, "Английский": "en"}

    def _cur_geom():
        try:
            w, h = root.winfo_width(), root.winfo_height()
            if w > 100 and h > 100:  # до первой разметки winfo возвращает 1
                state["window_size"] = f"{w}x{h}"
        except Exception:
            pass

    def save_cfg(*_):
        try:
            _cur_geom()
            CFG_PATH.write_text(json.dumps({
                "model": v_model.get(), "device": DEV_MAP[v_dev.get()],
                "language": v_lang.get() if v_lang.get() != "Авто" else "auto",
                "srt": v_srt.get(), "structured": v_struct.get(),
                "start_minimized": v_minimized.get(),
                "auto_process": v_auto.get(), "watch_dir": state["watch_dir"],
                "watching": state["watching"],
                "project_dir": v_project.get().strip(),
                "ai_enabled": v_ai_on.get(), "ai_command": v_ai_cmd.get(),
                "ai_mode": v_ai_mode.get(), "ai_region": v_ai_region.get(),
                "ai_model": v_ai_model.get().strip(),
                "oc_cli": v_oc_cli.get().strip(), "oc_model": v_oc_model.get().strip(),
                "oc_conn": v_oc_conn.get(), "oc_url": v_oc_url.get().strip(),
                "oc_variant": v_oc_variant.get().strip(),
                "go_model": v_go_model.get().strip(), "go_effort": v_go_effort.get().strip(),
                "ai_prompt": ai_prompt_text["v"], "whisper_prompt": whisper_prompt_text["v"],
                "measured": state["measured"],
                "last_dir": state.get("last_dir") or "",
                "window_size": state.get("window_size") or "1120x740",
                "settings_size": state.get("settings_size") or "760x820",
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
        win.minsize(720, 680)
        win.withdraw()  # не показывать, пока не займём итоговое положение (не «прыгает» из угла)
        dark_titlebar(win)

        # нижняя закреплённая полоса с кнопкой «Готово» — видна всегда
        bar_s = ttk.Frame(win)
        bar_s.pack(side="bottom", fill="x", padx=S["l"], pady=(0, S["m"]))

        # прокручиваемая область: содержимое настроек выше окна не обрезается
        outer = ttk.Frame(win)
        outer.pack(side="top", fill="both", expand=True, padx=S["l"], pady=(S["l"], 0))
        canvas = tk.Canvas(outer, bg=C["bg"], highlightthickness=0, bd=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        frame = ttk.Frame(canvas, style="Card.TFrame", padding=(S["xl"], S["l"]))
        _wid = canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(_wid, width=e.width))
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        if IS_MAC:
            def _wheel(e):
                canvas.yview_scroll(-1 * e.delta, "units")
            win.bind_all("<MouseWheel>", _wheel)
        else:
            def _wheel(e):
                canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
            win.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-3, "units"))
            win.bind_all("<Button-5>", lambda e: canvas.yview_scroll(3, "units"))
            win.bind_all("<MouseWheel>", _wheel)
        win.bind("<Destroy>", lambda e: (canvas.unbind_all("<MouseWheel>"),
                                         canvas.unbind_all("<Button-4>"),
                                         canvas.unbind_all("<Button-5>")) if e.widget is win else None)

        def close_settings():
            state["settings_size"] = f"{win.winfo_width()}x{win.winfo_height()}"
            ai_prompt_text["v"] = txt_prompt.get("1.0", "end").strip()
            whisper_prompt_text["v"] = txt_whisper.get("1.0", "end").strip()
            for _k in ("go_model_box_real", "upd_go_efforts", "refresh_go_state"):
                state[_k] = None  # виджеты уничтожаются — poll их трогать не должен
            save_cfg()
            win.destroy()

        win.bind("<Escape>", lambda e: close_settings())  # Esc закрывает диалог

        ttk.Button(bar_s, text="Готово", style="Accent.TButton",
                   command=lambda: close_settings()).pack(side="right")

        def scrolled_text(parent, height):
            """Текстовое поле с вертикальной прокруткой в тёмной теме."""
            holder = ttk.Frame(parent, style="Inner.TFrame")
            sb = ttk.Scrollbar(holder, orient="vertical")
            txt = tk.Text(holder, height=height, wrap="word", bg=C["panel2"], fg=C["fg"],
                          insertbackground=C["fg"], relief="flat", font=F["body"],
                          padx=8, pady=6, highlightthickness=0, selectbackground=C["accentSel"],
                          yscrollcommand=sb.set)
            sb.config(command=txt.yview)
            txt.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
            sb.pack(side="right", fill="y", pady=6, padx=(0, 6))
            return holder, txt

        ttk.Checkbutton(frame, text="Запускать программу вместе с Windows",
                        variable=v_autostart, style="Card.TCheckbutton",
                        command=on_autostart_toggle).pack(anchor="w")
        ttk.Checkbutton(frame, text="Запускать свёрнутой в трей (окно не показывать)",
                        variable=v_minimized, style="Card.TCheckbutton",
                        command=save_cfg).pack(anchor="w", pady=(S["xs"], 0))
        ttk.Separator(frame).pack(fill="x", pady=S["m"])

        # --- подсказка для распознавания ---
        ttk.Label(frame, text="Распознавание (Whisper)",
                  style="Section.TLabel").pack(anchor="w")
        ttk.Label(frame, style="CardMuted.TLabel", wraplength=620, justify="left",
                  text="Подсказка распознаванию — перечислите термины, названия сервисов и брендов "
                       "как они должны писаться (например: Kimi, Whisper, CUDA, Bitrix24). "
                       "Whisper подстроит написание под подсказку.").pack(anchor="w", pady=(S["xs"], S["xs"]))
        w_holder, txt_whisper = scrolled_text(frame, 4)
        w_holder.pack(fill="x", pady=(0, S["xs"]))
        txt_whisper.insert("1.0", whisper_prompt_text["v"])
        ttk.Separator(frame).pack(fill="x", pady=S["m"])

        # --- нейросеть для постобработки ---
        ttk.Label(frame, text="Обработка результата нейросетью",
                  style="Section.TLabel").pack(anchor="w")
        ttk.Checkbutton(frame, text="после распознавания отправлять текст нейросети",
                        variable=v_ai_on, style="Card.TCheckbutton",
                        command=save_cfg).pack(anchor="w", pady=(S["xs"], S["s"]))

        row = ttk.Frame(frame, style="Inner.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text="Способ:", style="CardMuted.TLabel").pack(side="left")
        MODES = ["Kimi API (ключ)", "OpenCode Go (ключ)", "OpenCode (CLI)", "Kimi CLI", "Ollama", "Своя команда"]
        cb_mode = ttk.Combobox(row, values=MODES, width=18, state="readonly")
        cb_mode.pack(side="left", padx=(S["s"], 0))
        cb_mode.set("Kimi API (ключ)" if v_ai_mode.get() == "api"
                    else ("OpenCode Go (ключ)" if v_ai_mode.get() == "go"
                          else ("OpenCode (CLI)" if v_ai_mode.get() == "opencode"
                                else ("Kimi CLI" if "kimi" in v_ai_cmd.get()
                                      else "Ollama" if "ollama" in v_ai_cmd.get() else "Своя команда"))))

        api_frame = ttk.Frame(frame, style="Inner.TFrame")
        ttk.Label(api_frame, text="Регион:", style="CardMuted.TLabel").grid(row=0, column=0, sticky="w")
        cb_region = ttk.Combobox(api_frame, textvariable=v_ai_region, state="readonly",
                                 values=["global", "mainland-cn"], width=15)
        cb_region.grid(row=0, column=1, sticky="w", padx=(S["s"], 0))
        ttk.Label(api_frame, text="API-ключ:", style="CardMuted.TLabel").grid(row=1, column=0, sticky="w", pady=(S["s"], 0))
        key_row = ttk.Frame(api_frame, style="Inner.TFrame")
        key_row.grid(row=1, column=1, sticky="w", padx=(S["s"], 0), pady=(S["s"], 0))
        ttk.Entry(key_row, textvariable=v_ai_key_new, width=30, show="•").pack(side="left")
        ttk.Button(key_row, text="Сохранить ключ", command=lambda: on_save_key()).pack(side="left", padx=(S["s"], 0))
        ttk.Button(key_row, text="Удалить", command=lambda: on_delete_key()).pack(side="left", padx=(S["s"], 0))
        lbl_key_state = ttk.Label(api_frame, style="CardMuted.TLabel")
        lbl_key_state.grid(row=2, column=1, sticky="w", padx=(S["s"], 0), pady=(S["xs"], 0))
        ttk.Label(api_frame, text="Модель:", style="CardMuted.TLabel").grid(row=3, column=0, sticky="w", pady=(S["s"], 0))
        model_row = ttk.Frame(api_frame, style="Inner.TFrame")
        model_row.grid(row=3, column=1, sticky="w", padx=(S["s"], 0), pady=(S["s"], 0))
        cb_kmodel = ttk.Combobox(model_row, textvariable=v_ai_model, values=KIMI_MODELS, width=28)
        cb_kmodel.pack(side="left")
        state["cb_kmodel"] = cb_kmodel
        ttk.Button(model_row, text="Обновить каталог", command=lambda: on_refresh_models()).pack(side="left", padx=(S["s"], 0))

        def refresh_key_state():
            if load_key():
                lbl_key_state.config(text="✓ ключ сохранён (в общих настройках не хранится)",
                                     foreground=C["green"])
            else:
                lbl_key_state.config(text="ключ не задан — вставьте и нажмите «Сохранить ключ»",
                                     foreground=C["yellow"])

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
            if load_key() and not messagebox.askyesno(
                    "Удалить API-ключ?",
                    "Ключ Kimi Code будет удалён с этого компьютера. ИИ-обработка перестанет работать, "
                    "пока вы не вставите ключ заново.", parent=win):
                return
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

        cmd_frame = ttk.Frame(frame, style="Inner.TFrame")
        ttk.Label(cmd_frame, text="Команда (текст уходит в stdin):", style="CardMuted.TLabel").pack(anchor="w")
        ttk.Entry(cmd_frame, textvariable=v_ai_cmd, width=64).pack(anchor="w", pady=(S["xs"], 0))
        v_ai_cmd.trace_add("write", save_cfg)

        # --- OpenCode: CLI (тот же фоновый сервер OpenCode, просто без ручных URL) ---
        # Скрытая возможность для продвинутых: "oc_conn":"server" + "oc_url" в
        # transcriber_config.json, пароль сервера — файлом private/oc_server_key.
        oc_frame = ttk.Frame(frame, style="Inner.TFrame")
        oc_frame.columnconfigure(2, weight=1)

        oc_cli_row = ttk.Frame(oc_frame, style="Inner.TFrame")
        ttk.Label(oc_frame, text="CLI:", style="CardMuted.TLabel").grid(row=0, column=0, sticky="w")
        oc_cli_row.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(S["s"], 0))
        ttk.Entry(oc_cli_row, textvariable=v_oc_cli).pack(side="left", fill="x", expand=True)
        ttk.Button(oc_cli_row, text="Найти", command=lambda: on_oc_find()).pack(side="left", padx=(S["s"], 0))
        v_oc_cli.trace_add("write", save_cfg)

        ttk.Label(oc_frame, text="Модель:", style="CardMuted.TLabel").grid(row=1, column=0, sticky="w", pady=(S["s"], 0))
        oc_model_row = ttk.Frame(oc_frame, style="Inner.TFrame")
        oc_model_row.grid(row=1, column=1, sticky="ew", padx=(S["s"], 0), pady=(S["s"], 0))
        oc_box = ttk.Combobox(oc_model_row, textvariable=v_oc_model, values=[], width=30)
        oc_box.pack(side="left")  # можно вписать свою модель или выбрать из списка
        state["oc_model_box"] = oc_box
        ttk.Button(oc_model_row, text="Список моделей", command=lambda: on_oc_models()).pack(side="left", padx=(S["s"], 0))
        v_oc_model.trace_add("write", save_cfg)

        ttk.Label(oc_frame, text="Рассуждение:", style="CardMuted.TLabel").grid(row=2, column=0, sticky="w", pady=(S["s"], 0))
        oc_var_row = ttk.Frame(oc_frame, style="Inner.TFrame")
        oc_var_row.grid(row=2, column=1, sticky="w", padx=(S["s"], 0), pady=(S["s"], 0))
        VARIANTS = ["", "none", "minimal", "low", "medium", "high", "max"]
        VAR_LABELS = {"": "по умолчанию", "none": "выключено", "minimal": "минимальная",
                      "low": "низкая", "medium": "средняя", "high": "высокая", "max": "максимальная"}
        cb_variant = ttk.Combobox(oc_var_row, textvariable=v_oc_variant,
                                  values=[""] + [VAR_LABELS[v] for v in VARIANTS[1:]],
                                  state="readonly", width=16)
        cb_variant.pack(side="left")
        lbl_variant = ttk.Label(oc_var_row, text="", style="CardMuted.TLabel")
        lbl_variant.pack(side="left", padx=(S["s"], 0))
        v_oc_variant.trace_add("write", lambda *_: (save_cfg(), upd_variant_label()))

        def upd_variant_label():
            i = cb_variant.current()
            real = VARIANTS[i] if i >= 0 else ""
            if real and "#" in v_oc_model.get():
                lbl_variant.config(text="модель уже содержит #variant — глубина из модели")
            elif real:
                lbl_variant.config(text="итого: " + (v_oc_model.get() or "модель") + "#" + real)
            else:
                lbl_variant.config(text="")
        cb_variant.set(VAR_LABELS.get(v_oc_variant.get().strip(), "по умолчанию"))
        upd_variant_label()

        lbl_oc_state = ttk.Label(oc_frame, style="CardMuted.TLabel", wraplength=560, justify="left")
        lbl_oc_state.grid(row=3, column=1, columnspan=2, sticky="w", padx=(S["s"], 0), pady=(S["xs"], 0))

        def refresh_oc_state():
            p = v_oc_cli.get().strip()
            if not p:
                lbl_oc_state.config(text="CLI не найден — установите OpenCode или укажите путь вручную",
                                   foreground=C["yellow"])
            elif not Path(p).is_file():
                lbl_oc_state.config(text="файл не найден — проверьте путь", foreground=C["yellow"])
            else:
                lbl_oc_state.config(text="✓ " + p, foreground=C["green"])
            upd_variant_label()

        def on_oc_find():
            p = find_opencode_cli()
            if p:
                v_oc_cli.set(p)
                status("OpenCode найден: " + p)
            else:
                status("OpenCode не найден — впишите путь к opencode вручную")
            refresh_oc_state()

        def on_oc_models():
            if v_oc_conn.get() == "server":
                status("Запрашиваю каталог моделей OpenCode (server)…")
                base = v_oc_url.get().strip().rstrip("/")
                key = load_oc_key()
                def job():
                    try:
                        import urllib.request
                        import base64
                        b = base if base.startswith("http") else "http://" + base
                        hdr = {}
                        if key:
                            tok = base64.b64encode(("opencode:" + key).encode()).decode()
                            hdr["Authorization"] = "Basic " + tok
                        r = urllib.request.Request(b + "/api/provider", headers=hdr)
                        with urllib.request.urlopen(r, timeout=30) as resp:
                            data = json.loads(resp.read()).get("data", [])
                        ids = []
                        for pr in data:
                            for m in (pr.get("models") or {}).values() if isinstance(pr.get("models"), dict) else []:
                                mid = m.get("id") or ""
                                pid = pr.get("id") or ""
                                if mid and pid:
                                    ids.append(pid + "/" + mid)
                        q.put(("oc_models", ids or None, None if ids else "пустой каталог"))
                    except Exception as e:
                        q.put(("oc_models", None, str(e)[:120]))
                threading.Thread(target=job, daemon=True).start()
                return
            if not Path(v_oc_cli.get().strip()).is_file():
                status("Сначала укажите путь к CLI OpenCode")
                return
            status("Запрашиваю список моделей OpenCode…")
            cli = v_oc_cli.get().strip()
            def job():
                try:
                    proc = subprocess.run([cli, "models"], capture_output=True, timeout=120,
                                          encoding="utf-8", errors="replace")
                    lines = []
                    for ln in (proc.stdout or "").splitlines():
                        ln = ln.strip()
                        if ln and "/" in ln:
                            lines.append(ln.split()[0])
                    q.put(("oc_models", lines or None, None if lines else (proc.stderr or "")[:120]))
                except Exception as e:
                    q.put(("oc_models", None, str(e)[:120]))
            threading.Thread(target=job, daemon=True).start()

        refresh_oc_state()

        # --- OpenCode Go: подписка через официальный CLI (контракт как в эталонном проекте) ---
        go_frame = ttk.Frame(frame, style="Inner.TFrame")
        ttk.Label(go_frame, text="API-ключ Go:", style="CardMuted.TLabel").grid(row=0, column=0, sticky="w")
        go_key_row = ttk.Frame(go_frame, style="Inner.TFrame")
        go_key_row.grid(row=0, column=1, sticky="ew", padx=(S["s"], 0))
        go_frame.columnconfigure(1, weight=1)
        ttk.Entry(go_key_row, textvariable=v_go_key_new, show="•", width=28).pack(side="left")
        ttk.Button(go_key_row, text="Сохранить", command=lambda: on_go_save_key()).pack(side="left", padx=(S["s"], 0))
        ttk.Button(go_key_row, text="Забыть", command=lambda: on_go_forget()).pack(side="left", padx=(S["s"], 0))
        lbl_go_key = ttk.Label(go_frame, style="CardMuted.TLabel")
        lbl_go_key.grid(row=1, column=1, sticky="w", padx=(S["s"], 0), pady=(S["xs"], 0))
        ttk.Label(go_frame, text="Модель:", style="CardMuted.TLabel").grid(row=2, column=0, sticky="w", pady=(S["s"], 0))
        go_model_row = ttk.Frame(go_frame, style="Inner.TFrame")
        go_model_row.grid(row=2, column=1, sticky="ew", padx=(S["s"], 0), pady=(S["s"], 0))
        go_mbox = ttk.Combobox(go_model_row, textvariable=v_go_model, values=GO_MODELS_DEFAULT, width=26)
        go_mbox.pack(side="left")
        ttk.Button(go_model_row, text="Список моделей", command=lambda: on_go_models()).pack(side="left", padx=(S["s"], 0))
        ttk.Button(go_model_row, text="Проверить связь", command=lambda: on_go_verify()).pack(side="left", padx=(S["s"], 0))
        ttk.Label(go_frame, text="Рассуждение:", style="CardMuted.TLabel").grid(row=3, column=0, sticky="w", pady=(S["s"], 0))
        go_eff_row = ttk.Frame(go_frame, style="Inner.TFrame")
        go_eff_row.grid(row=3, column=1, sticky="w", padx=(S["s"], 0), pady=(S["s"], 0))
        go_ebox = ttk.Combobox(go_eff_row, values=["по умолчанию модели"], state="readonly", width=20)
        go_ebox.pack(side="left")
        go_ebox.set("по умолчанию модели")
        lbl_go_eff = ttk.Label(go_eff_row, text="", style="CardMuted.TLabel")
        lbl_go_eff.pack(side="left", padx=(S["s"], 0))
        lbl_go_state = ttk.Label(go_frame, style="CardMuted.TLabel", wraplength=560, justify="left")
        lbl_go_state.grid(row=4, column=1, sticky="w", padx=(S["s"], 0), pady=(S["xs"], 0))
        v_go_model.trace_add("write", lambda *_: (save_cfg(), upd_go_efforts()))
        v_go_effort.trace_add("write", save_cfg)

        def refresh_go_state():
            k = load_go_key()
            if k and state.get("go_verified"):
                lbl_go_key.config(text="✓ ключ сохранён, связь с моделью проверена", foreground=C["green"])
            elif k:
                lbl_go_key.config(text="ключ сохранён — нажмите «Проверить связь»", foreground=C["yellow"])
            else:
                lbl_go_key.config(text="ключ не задан — возьмите его в opencode.ai/console (Go → API key)",
                                  foreground=C["muted"])
            lbl_go_state.config(text="запросы идут напрямую через API OpenCode Go — OpenCode CLI не нужен",
                               foreground=C["muted"])

        def upd_go_efforts(*_):
            mid = v_go_model.get().strip().split("/", 1)[-1]
            efforts = []
            for m in state.get("go_catalog") or []:
                if m["id"] == mid:
                    efforts = m.get("efforts") or []
                    break
            go_ebox.config(values=["по умолчанию модели"] + efforts)
            cur = v_go_effort.get() or "по умолчанию модели"
            go_ebox.set(cur if cur in (["по умолчанию модели"] + efforts) else "по умолчанию модели")
            lbl_go_eff.config(text="" if efforts else "у этой модели уровней нет")

        def on_go_effort_sel(*_):
            sel = go_ebox.get()
            v_go_effort.set("" if sel.startswith("по умолчанию") else sel)
        go_ebox.bind("<<ComboboxSelected>>", on_go_effort_sel)

        def on_go_save_key():
            k = v_go_key_new.get().strip()
            if not k:
                status("Вставьте ключ из opencode.ai/console (раздел Go)")
                return
            try:
                save_go_key(k)
                v_go_key_new.set("")
                state["go_verified"] = False
                refresh_go_state()
                status("Ключ OpenCode Go сохранён (доступ к файлу только у вас)")
                on_go_models()  # подтянем каталог с реальными уровнями
            except Exception as e:
                log("сохранение ключа Go:", e)
                status("Не удалось сохранить ключ: " + str(e)[:80])

        def on_go_forget():
            if load_go_key() and not messagebox.askyesno(
                    "Забыть ключ OpenCode Go?",
                    "Ключ будет удалён с этого компьютера; обработка через Go перестанет работать, "
                    "пока вы не вставите ключ заново.", parent=win):
                return
            delete_go_key()
            v_go_key_new.set("")
            state["go_verified"] = False
            refresh_go_state()
            status("Ключ OpenCode Go удалён")

        def on_go_models():
            key = load_go_key()
            status("Загружаю каталог моделей OpenCode Go…")
            def job():
                try:
                    q.put(("go_catalog", go_fetch_catalog(key), None))
                except RuntimeError as e:
                    q.put(("go_catalog", None, str(e)))
                except Exception as e:
                    q.put(("go_catalog", None, str(e)[:120]))
            threading.Thread(target=job, daemon=True).start()

        def on_go_verify():
            if not load_go_key():
                status("Сначала сохраните ключ OpenCode Go")
                return
            status("Проверяю связь с моделью (без данных проекта)…")
            mid = v_go_model.get().strip()
            def job():
                try:
                    out = run_opencode_go(mid, "", 'Ответь ровно: {"ok":true}', "")
                    q.put(("go_verify", '"ok"' in out.lower().replace(" ", ""), None))
                except RuntimeError as e:
                    q.put(("go_verify", False, str(e)))
                except Exception as e:
                    q.put(("go_verify", False, str(e)[:150]))
            threading.Thread(target=job, daemon=True).start()

        refresh_go_state()
        upd_go_efforts()
        state["go_model_box_real"] = go_mbox
        state["upd_go_efforts"] = upd_go_efforts
        state["refresh_go_state"] = refresh_go_state

        def on_mode_change(*_):
            m = cb_mode.get()
            if m == "Kimi API (ключ)":
                v_ai_mode.set("api")
                cmd_frame.pack_forget()
                oc_frame.pack_forget()
                go_frame.pack_forget()
                api_frame.pack(fill="x", pady=(8, 0))
            elif m == "OpenCode Go (ключ)":
                v_ai_mode.set("go")
                api_frame.pack_forget()
                cmd_frame.pack_forget()
                oc_frame.pack_forget()
                go_frame.pack(fill="x", pady=(8, 0))
            elif m == "OpenCode (CLI)":
                v_ai_mode.set("opencode")
                api_frame.pack_forget()
                cmd_frame.pack_forget()
                go_frame.pack_forget()
                oc_frame.pack(fill="x", pady=(8, 0))
            else:
                v_ai_mode.set("cli")
                api_frame.pack_forget()
                oc_frame.pack_forget()
                go_frame.pack_forget()
                cmd_frame.pack(fill="x", pady=(8, 0))
                v_ai_cmd.set(AI_PRESETS.get(m, v_ai_cmd.get()))
            save_cfg()

        cb_mode.bind("<<ComboboxSelected>>", on_mode_change)
        on_mode_change()

        ttk.Label(frame, style="CardMuted.TLabel", justify="left", wraplength=640,
                  text="Ответ сохранится в файл <имя>.ai.txt и переедет в папку проекта вместе с остальными"
                  ).pack(anchor="w", pady=(S["s"], S["s"]))

        ttk.Label(frame, text="Промпт:", style="CardKicker.TLabel").pack(anchor="w")
        p_holder, txt_prompt = scrolled_text(frame, 4)
        p_holder.pack(fill="x", pady=(S["xs"], S["m"]))
        txt_prompt.insert("1.0", ai_prompt_text["v"])

        win.protocol("WM_DELETE_WINDOW", close_settings)
        _sg = (state.get("settings_size") or "760x820").lower().split("x", 1)
        try:
            _sw, _sh = int(_sg[0]), int(_sg[1])
        except Exception:
            _sw, _sh = 760, 820
        center_on_root(win, _sw, _sh)
        win.deiconify()  # окно появляется сразу по центру
        win.focus_set()

    # ----- шапка -----
    head = ttk.Frame(root, padding=(S["xl"], S["l"], S["xl"], S["xs"]))
    head.pack(fill="x")
    brand = ttk.Frame(head)
    brand.pack(side="left")
    ttk.Label(brand, text="Транскрибатор", style="Title.TLabel").pack(anchor="w")
    ttk.Label(brand, text="аудио и видео → текст · распознавание локально",
              style="Sub.TLabel").pack(anchor="w")
    ttk.Button(head, text="Настройки", style="Ghost.TButton",
               command=lambda: open_settings()).pack(side="right")
    if IS_WIN:
        ttk.Button(head, text="В трей", width=7, style="Ghost.TButton",
                   command=lambda: root.withdraw()).pack(side="right", padx=S["s"])
    lbl_engine = ttk.Label(head, text="", style="Pill.TLabel")  # появится, когда движок определён

    # ----- модель и формат вывода -----
    card1 = ttk.Frame(root, style="Card.TFrame", padding=(S["l"], S["m"]))
    card1.pack(fill="x", padx=S["xl"], pady=(S["s"], S["s"]))
    ttk.Label(card1, text="МОДЕЛЬ", style="CardKicker.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(card1, text="УСТРОЙСТВО", style="CardKicker.TLabel").grid(row=0, column=1, sticky="w", padx=(S["m"], 0))
    ttk.Label(card1, text="ЯЗЫК", style="CardKicker.TLabel").grid(row=0, column=2, sticky="w", padx=(S["m"], 0))
    ttk.Label(card1, text="ФОРМАТ ВЫВОДА", style="CardKicker.TLabel").grid(row=0, column=4, sticky="w", padx=(S["xl"], 0))
    cb_model = ttk.Combobox(card1, textvariable=v_model, values=MODEL_LIST, width=18, state="readonly")
    cb_model.grid(row=1, column=0, sticky="w", pady=(S["xs"], 0))
    cb_dev = ttk.Combobox(card1, textvariable=v_dev, values=list(DEV_MAP), width=12, state="readonly")
    cb_dev.grid(row=1, column=1, sticky="w", padx=(S["m"], 0), pady=(S["xs"], 0))
    cb_lang = ttk.Combobox(card1, textvariable=v_lang, values=list(LANG_MAP), width=11, state="readonly")
    cb_lang.grid(row=1, column=2, sticky="w", padx=(S["m"], 0), pady=(S["xs"], 0))
    ttk.Separator(card1, orient="vertical").grid(row=0, column=3, rowspan=2, sticky="ns", padx=(S["l"], S["xl"]))
    ttk.Checkbutton(card1, text="субтитры .srt", variable=v_srt,
                    style="Card.TCheckbutton", command=save_cfg).grid(row=1, column=4, sticky="w", pady=(S["xs"], 0))
    ttk.Checkbutton(card1, text=".txt с таймкодами (структурированный)", variable=v_struct,
                    style="Card.TCheckbutton", command=save_cfg).grid(row=2, column=4, sticky="w", pady=(S["xs"], 0))
    cb_dev.bind("<<ComboboxSelected>>", save_cfg)
    cb_lang.bind("<<ComboboxSelected>>", save_cfg)

    # ----- добавление файлов -----
    card2 = ttk.Frame(root, style="Card.TFrame", padding=(S["l"], S["m"]))
    card2.pack(fill="x", padx=S["xl"], pady=(0, S["s"]))
    card2.columnconfigure(2, weight=1)

    def dialog_dir():
        """Стартовая папка диалогов: последняя выбранная, слежение, проект или «Видео»."""
        for cand in (state.get("last_dir"), state.get("watch_dir"), v_project.get().strip(),
                     str(Path.home() / "Videos")):
            try:
                if cand and Path(cand).is_dir():
                    return str(cand)
            except OSError:
                pass
        return str(Path.home())

    def add_files(paths, report=True):
        added, skipped, dup = 0, 0, 0
        skipped_ext = set()
        for p in paths:
            p = Path(p)
            if not p.exists():
                skipped += 1
                continue
            if p.suffix.lower() not in MEDIA_EXT:
                skipped += 1
                if p.suffix:
                    skipped_ext.add(p.suffix.lower())
                continue
            key = str(p)
            if tree.exists(key):
                dup += 1
                continue
            dur = probe_duration(p)
            est = estimate(v_model.get(), resolved_dev(), dur)
            tree.insert("", "end", iid=key, values=(
                p.name, fmt_dur(dur), "≈ " + fmt_time(est) if est else "?", "ожидает", ""),
                tags=("wait",))
            added += 1
        if added:
            update_total()
        if report:
            if added:
                extra = []
                if skipped:
                    extra.append("пропущено: " + str(skipped))
                if dup:
                    extra.append("уже в очереди: " + str(dup))
                status("Добавлено файлов: " + str(added) +
                       (" (" + ", ".join(extra) + ")" if extra else ""))
            elif dup:
                status("Эти файлы уже есть в очереди")
            elif skipped:
                exts = ", ".join(sorted(skipped_ext)) or "неизвестные"
                status("Файлы не добавлены — неподдерживаемый формат: " + exts)
            else:
                status("Файлы не выбраны")
        return added, skipped, dup

    def pick_files():
        paths = filedialog.askopenfilenames(
            parent=root,
            title="Выберите аудио/видео файлы",
            initialdir=dialog_dir(),
            filetypes=[("Все файлы", "*.*"),
                       ("Аудио и видео", " ".join("*" + e for e in sorted(MEDIA_EXT)))])
        if paths:
            state["last_dir"] = str(Path(paths[0]).parent)
            save_cfg()
            add_files(paths)

    def pick_folder_files():
        d = filedialog.askdirectory(parent=root, title="Папка с файлами для обработки",
                                    initialdir=dialog_dir())
        if d:
            state["last_dir"] = d
            save_cfg()
            add_files(sorted(Path(d).iterdir()))

    row_add = ttk.Frame(card2, style="Inner.TFrame")
    row_add.grid(row=0, column=0, columnspan=3, sticky="ew")
    b_files = ttk.Button(row_add, text="Выбрать файлы…", width=17, command=pick_files)
    b_folder = ttk.Button(row_add, text="Обработать папку…", width=17, command=pick_folder_files)
    b_files.pack(side="left")
    b_folder.pack(side="left", padx=(S["s"], 0))
    ttk.Label(row_add, text="можно перетащить файлы прямо в окно",
              style="CardMuted.TLabel").pack(side="right")

    # ----- слежение и папка проекта -----
    ttk.Label(card2, text="СЛЕЖЕНИЕ ЗА ПАПКОЙ", style="CardKicker.TLabel").grid(row=1, column=0, sticky="w", pady=(S["m"], 0))
    row_watch = ttk.Frame(card2, style="Inner.TFrame")
    row_watch.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(S["xs"], 0))

    ent_watch = ttk.Entry(row_watch, textvariable=v_watch)
    ent_watch.pack(side="left", fill="x", expand=True)

    def pick_watch():
        d = filedialog.askdirectory(title="Папка для слежения")
        if d:
            v_watch.set(d)
            save_cfg()

    def toggle_watch():
        if state["watching"]:
            state["watching"] = False
            btn_watch.config(text="▶  Возобновить")
            status("Слежение остановлено")
            save_cfg()
            return
        d = Path(v_watch.get().strip())
        if not d.is_dir():
            messagebox.showwarning("Транскрибатор", "Выберите существующую папку для слежения.")
            return
        state["watch_dir"] = str(d)
        state["seen"] = {str(p) for p in d.iterdir() if p.is_file()}
        state["watching"] = True
        btn_watch.config(text="Остановить")
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

    ttk.Button(row_watch, text="Обзор…", width=8, command=pick_watch).pack(side="left", padx=(S["s"], S["s"]))
    btn_watch = ttk.Button(row_watch, text="▶  Начать слежение", width=19, command=toggle_watch)
    btn_watch.pack(side="left")
    ttk.Checkbutton(row_watch, text="обрабатывать автоматически", variable=v_auto,
                    style="Card.TCheckbutton", command=save_cfg).pack(side="left", padx=S["m"])

    ttk.Label(card2, text="ПАПКА ПРОЕКТА (КУДА ПЕРЕНЕСТИ РЕЗУЛЬТАТЫ)", style="CardKicker.TLabel").grid(row=3, column=0, sticky="w", pady=(S["m"], 0))
    row_proj = ttk.Frame(card2, style="Inner.TFrame")
    row_proj.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(S["xs"], 0))
    ent_proj = ttk.Entry(row_proj, textvariable=v_project)
    ent_proj.pack(side="left", fill="x", expand=True)

    def pick_project():
        d = filedialog.askdirectory(title="Куда переносить файл и результаты после обработки")
        if d:
            v_project.set(d)
            save_cfg()

    def clear_project():
        v_project.set("")
        save_cfg()

    ttk.Button(row_proj, text="Обзор…", width=8, command=pick_project).pack(side="left", padx=(S["s"], S["s"]))
    ttk.Button(row_proj, text="✕  Не переносить", command=clear_project).pack(side="left")
    ttk.Label(row_proj, style="CardMuted.TLabel",
              text="пусто — файлы останутся рядом с исходником"
              ).pack(side="left", padx=S["m"])

    # ----- нижняя панель (пакуем первой, чтобы не обрезалась) -----
    bottom = ttk.Frame(root, padding=(S["xl"], S["s"], S["xl"], S["l"]))
    bottom.pack(side="bottom", fill="x")
    btns = ttk.Frame(bottom)   # обе строки действий — в одной сетке из 5 равных колонок
    btns.pack(fill="x")
    for _c in range(5):
        btns.columnconfigure(_c, weight=1, uniform="act", pad=0)

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
            open_file(txt)
        else:
            messagebox.showinfo("Транскрибатор", "Для этого файла результат ещё не готов.")

    def open_ai():
        """Открыть конспект нейросети (.ai.txt) для выбранного файла."""
        sel = tree.selection()
        if not sel:
            messagebox.showinfo("Транскрибатор", "Выберите обработанный файл в списке.")
            return
        txt = state["results"].get(sel[0])
        ai = Path(txt).with_suffix(".ai.txt") if txt else None
        if ai and ai.exists():
            open_file(ai)
        else:
            messagebox.showinfo("Транскрибатор",
                                "Конспекта пока нет — включите «отправлять текст нейросети» в настройках.")

    def open_folder():
        """Открыть папку, где лежит результат выбранного файла."""
        sel = tree.selection()
        if not sel:
            messagebox.showinfo("Транскрибатор", "Выберите файл в списке.")
            return
        txt = state["results"].get(sel[0])
        folder = Path(txt).parent if txt else Path(sel[0]).parent
        if folder.exists():
            open_file(folder)
        else:
            messagebox.showinfo("Транскрибатор", "Папка не найдена: " + str(folder))

    # обе строки действий — в одной сетке из 5 равных колонок (uniform), поэтому
    # кнопки верхнего и нижнего рядов имеют строго одинаковую ширину
    btn_start = ttk.Button(btns, text="▶  Начать обработку", style="Accent.TButton",
                           command=lambda: start_processing(auto=False))
    btn_start.grid(row=0, column=0, sticky="ew", padx=(S["xs"], S["xs"]), pady=(0, S["xs"]))
    btn_stop = ttk.Button(btns, text="■ Стоп после текущего", style="Peer.TButton",
                          command=lambda: state.update(stop=True))
    btn_stop.grid(row=0, column=1, sticky="ew", padx=(S["xs"], S["xs"]), pady=(0, S["xs"]))
    def remove_selected():
        """Убрать выбранные строки из очереди (только те, что ещё не в работе)."""
        removed, skipped = 0, 0
        for iid in tree.selection():
            vals = tree.item(iid, "values")
            if vals[3] == "обрабатываю…":
                skipped += 1
                continue
            state["seen"].add(str(iid))  # слежение не должно добавить его снова
            state["results"].pop(iid, None)
            tree.delete(iid)
            removed += 1
        if removed:
            update_total()
            status(f"Убрано из очереди: {removed}" + (f", пропущено (уже в работе): {skipped}" if skipped else ""))
        elif skipped:
            status("Файл уже обрабатывается — дождитесь завершения или нажмите «Стоп»")

    u_remove = ttk.Button(btns, text="✕  Убрать из очереди", command=remove_selected)
    u_clean = ttk.Button(btns, text="Очистить готовые", command=clear_done)
    u_result = ttk.Button(btns, text="Открыть результат", command=open_result)
    u_ai = ttk.Button(btns, text="Открыть конспект", command=open_ai)
    u_folder = ttk.Button(btns, text="Папка с результатом", command=open_folder)
    for _i, _b in enumerate((u_remove, u_clean, u_result, u_ai, u_folder)):
        _b.grid(row=1, column=_i, sticky="ew", padx=(S["xs"], S["xs"]))

    bar = ttk.Progressbar(bottom, mode="determinate", style="accent.Horizontal.TProgressbar")
    bar.pack(fill="x", pady=(S["l"], S["s"]))
    statusrow = ttk.Frame(bottom)
    statusrow.pack(fill="x")
    lbl_status = ttk.Label(statusrow, text="Готов к работе", style="Status.TLabel")
    lbl_status.pack(side="left")
    lbl_total = ttk.Label(statusrow, text="", style="Muted.TLabel", font=F["mono"])
    lbl_total.pack(side="right")

    # ----- очередь -----
    qcard = ttk.Frame(root, style="Card.TFrame", padding=(S["s"], S["s"]))
    qcard.pack(fill="both", expand=True, padx=S["xl"], pady=(0, S["s"]))
    cols = ("file", "dur", "est", "status", "real")
    tree = ttk.Treeview(qcard, columns=cols, show="headings", height=10)
    tree.heading("file", text="ФАЙЛ")
    tree.heading("dur", text="ДЛИТЕЛЬНОСТЬ")
    tree.heading("est", text="ОЦЕНКА ВРЕМЕНИ")
    tree.heading("status", text="СТАТУС")
    tree.heading("real", text="ФАКТИЧЕСКИ")
    tree.column("file", width=380, minwidth=200, anchor="w", stretch=True)
    tree.column("dur", width=135, anchor="center", stretch=False)
    tree.column("est", width=170, anchor="center", stretch=False)
    tree.column("status", width=240, anchor="w", stretch=True)
    tree.column("real", width=125, anchor="center", stretch=False)
    tree.tag_configure("done", foreground=C["green"])
    tree.tag_configure("busy", foreground=C["accent2"])
    tree.tag_configure("error", foreground=C["red"])
    tree.tag_configure("wait", foreground=C["muted"])
    tree.pack(fill="both", expand=True)

    def on_double(event):
        sel = tree.selection()
        if sel and sel[0] in state["results"]:
            open_file(state["results"][sel[0]])

    tree.bind("<Double-1>", on_double)
    tree.bind("<Return>", lambda e: on_double(e))
    tree.bind("<Delete>", lambda e: remove_selected())

    # ----- перетаскивание файлов в окно (Windows) -----
    # WndProc только складывает пути в очередь, разбирает их poll().
    if IS_WIN:
        try:
            hook_dropfiles_tree(root)
        except Exception as e:
            log("drag&drop:", e)

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
        lbl_total.config(text=("В очереди: ≈ " + fmt_time(total)) if have
                         else "Очередь пуста — добавьте файлы или включите слежение")

    # ----- обработка -----
    def start_processing(auto=False):
        if state["busy"]:
            status("Обработка уже идёт — дождитесь завершения или нажмите «Стоп после текущего»")
            return
        pending = [iid for iid in tree.get_children() if tree.item(iid, "values")[3] == "ожидает"]
        if not pending:
            if not tree.get_children():
                status("Очередь пуста — добавьте файлы кнопкой «Выбрать файлы…» "
                       "или перетащите их в окно")
            else:
                status("Нет файлов в статусе «ожидает» — они уже обработаны или находятся в работе")
            return
        # фиксируем настройки до запуска потока (tkinter-переменные из потока читать нельзя)
        opts = {
            "model": v_model.get(), "device": DEV_MAP[v_dev.get()],
            "lang": LANG_MAP[v_lang.get()], "srt": v_srt.get(),
            "structured": v_struct.get(), "project_dir": v_project.get().strip(),
            "ai_on": v_ai_on.get(), "ai_cmd": v_ai_cmd.get(), "ai_prompt": ai_prompt_text["v"],
            "ai_mode": v_ai_mode.get(), "ai_region": v_ai_region.get(),
            "ai_model": v_ai_model.get().strip(),
            "oc_cli": v_oc_cli.get(), "oc_model": v_oc_model.get(),
            "oc_conn": v_oc_conn.get(), "oc_url": v_oc_url.get().strip(),
            "oc_key": load_oc_key(), "oc_variant": v_oc_variant.get().strip(),
            "go_model": v_go_model.get().strip(), "go_effort": v_go_effort.get().strip(),
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
                        elif opts["ai_mode"] == "go":
                            lvl = "#" + opts["go_effort"] if opts["go_effort"] else ""
                            q.put(("status", f"OpenCode Go ({opts['go_model']}{lvl}): {p.name}"))
                            try:
                                ai_out = run_opencode_go(opts["go_model"],
                                                         opts["go_effort"], opts["ai_prompt"], plain_text)
                            except RuntimeError as e:
                                ai_out = None
                                msg = str(e)
                                q.put(("status", "OpenCode Go: " + msg))
                                if "402" in msg or "429" in msg or "лимит" in msg or "подписк" in msg:
                                    state["stop"] = True  # исчерпан тариф — нет смысла жечь оставшуюся очередь
                        elif opts["ai_mode"] == "opencode" and opts["oc_conn"] == "server" and opts["oc_url"].strip():
                            model_full = opts["oc_model"].strip()
                            if opts["oc_variant"] and "#" not in model_full:
                                model_full = model_full + "#" + opts["oc_variant"]
                            q.put(("status", f"OpenCode server ({model_full or 'модель по умолчанию'}): {p.name}"))
                            ai_out = run_opencode_server(opts["oc_url"], opts["oc_key"],
                                                         model_full, opts["ai_prompt"], plain_text)
                        elif opts["ai_mode"] == "opencode" and opts["oc_cli"].strip():
                            model_full = opts["oc_model"].strip()
                            if opts["oc_variant"] and "#" not in model_full:
                                model_full = model_full + "#" + opts["oc_variant"]
                            q.put(("status", f"OpenCode ({model_full or 'модель по умолчанию'}): {p.name}"))
                            ai_out = run_opencode(opts["oc_cli"].strip(), model_full,
                                                  opts["ai_prompt"], plain_text)
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
            msg = str(e)
            low = msg.lower()
            if isinstance(e, ImportError):
                msg = "не удалось загрузить движок распознавания: " + msg
            elif any(k in low for k in ("connection", "connect", "timed out", "timeout",
                                         "download", "huggingface", "certificate", "ssl")):
                msg += " — проверьте интернет: при первом запуске модель скачивается"
            q.put(("status", "Ошибка запуска: " + msg))
        finally:
            q.put(("idle",))

    # ----- трей -----
    tray_icon = {"obj": None}

    def start_tray():
        if IS_MAC:
            return  # pystray на macOS требует Cocoa-цикл в главном потоке — трей отключаем,
                     # приложение живёт в Dock, закрытие окна = выход
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
        win.withdraw()  # появится сразу по центру, без «прыжка» из левого верхнего угла
        dark_titlebar(win)
        win.bind("<Escape>", lambda e: win.destroy())
        frame = ttk.Frame(win, style="Card.TFrame", padding=(S["xl"], S["l"]))
        frame.pack(fill="both", expand=True, padx=S["l"], pady=S["l"])

        ttk.Label(frame, text=f"Обработано файлов: {len(items)}",
                  style="Section.TLabel", font=F["title"]).pack(anchor="w")
        lst = tk.Listbox(frame, height=5, bg=C["panel2"], fg=C["fg"], relief="flat",
                         highlightthickness=0, font=F["body"], activestyle="none",
                         selectbackground=C["accentSel"], selectforeground="#FFFFFF")
        for it in items:
            lst.insert("end", "•  " + it["name"])
        lst.pack(fill="x", pady=(S["m"], S["s"]))
        lbl_loc = ttk.Label(frame, style="CardMuted.TLabel", wraplength=520, justify="left")
        lbl_loc.pack(anchor="w", pady=(0, S["s"]))

        def refresh_loc():
            lbl_loc.config(text="Сейчас файлы лежат: " +
                           str(Path(next(iter(items[0]["files"].values()))).parent),
                           font=F["mono"])

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
            open_file(str(Path(next(iter(items[0]["files"].values()))).parent))

        ttk.Button(frame, text="Оставить как есть", command=win.destroy).pack(side="bottom", anchor="e")
        rowb = ttk.Frame(frame, style="Inner.TFrame")
        rowb.pack(fill="x", pady=(S["s"], 0))
        ttk.Button(rowb, text="Переместить в папку…", style="Accent.TButton",
                   command=move_all).pack(side="left")
        ttk.Button(rowb, text="Открыть расположение", command=open_loc).pack(side="left", padx=(S["s"], 0))
        center_on_root(win, 600, 380)
        win.deiconify()
        win.focus_set()

    # ----- события из потоков -----
    def poll():
        quit_after = False
        try:
            # файлы, принесённые перетаскиванием (WndProc пишет их в очередь)
            for files in drain_dropped_files():
                add_files(files)
            while True:
                ev = q.get_nowait()
                kind = ev[0]
                if kind == "add":
                    add_files([ev[1]], report=False)
                    if v_auto.get():
                        start_processing(auto=True)
                elif kind == "row":
                    vals = list(tree.item(ev[1], "values"))
                    vals[3] = ev[2]
                    tree.item(ev[1], values=vals, tags=(row_tag(ev[2]),))
                    if str(ev[2]).startswith("ошибка"):
                        status("Файл не обработан — подробности в transcriber.log")
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
                elif kind == "go_catalog":
                    _, models, err = ev
                    if models:
                        state["go_catalog"] = models
                        ids = [m["id"] for m in models]
                        try:
                            box = state.get("go_model_box_real")
                            if box is not None and box.winfo_exists():
                                box.config(values=ids)
                                if v_go_model.get() not in ids:
                                    v_go_model.set(ids[0])
                            upd = state.get("upd_go_efforts")
                            if callable(upd):
                                upd()
                        except tk.TclError:
                            pass  # окно настроек уже закрыто — каталог останется в state
                        status("Каталог OpenCode Go: " + str(len(ids)) + " моделей")
                    elif err:
                        status("Каталог Go: " + err)
                elif kind == "go_verify":
                    ok, err = ev[1], ev[2]
                    state["go_verified"] = bool(ok)
                    if ok:
                        status("OpenCode Go: связь с моделью подтверждена ✓")
                    else:
                        status("OpenCode Go: " + (err or "проверка не прошла"))
                    try:
                        rf = state.get("refresh_go_state")
                        if callable(rf):
                            rf()
                    except tk.TclError:
                        pass
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
                elif kind == "oc_models":
                    _, models, err = ev
                    if models:
                        oc_box = state.get("oc_model_box")
                        if oc_box is not None and oc_box.winfo_exists():
                            oc_box.config(values=models)
                        status("OpenCode: доступных моделей — " + str(len(models)))
                    else:
                        status("Список моделей OpenCode не получен: " + (err or "?"))
                elif kind == "engine":
                    lbl_engine.config(text="движок: видеокарта (CUDA)" if ev[1] == "cuda" else "движок: процессор")
                    if not lbl_engine.winfo_ismapped():
                        lbl_engine.pack(side="right", padx=S["l"])
                elif kind == "toggle_window":
                    toggle_window()
                elif kind == "quit":
                    on_close()
                    quit_after = True
                    break
                elif kind == "idle":
                    state["busy"] = False
                    btn_start.config(state="normal")
                    update_total()
                    # если за время обработки в очередь упали новые файлы — продолжить автоматически
                    pending_left = [iid for iid in tree.get_children()
                                    if tree.item(iid, "values")[3] == "ожидает"]
                    if pending_left and v_auto.get() and not state["stop"]:
                        root.after(300, lambda: start_processing(auto=True))
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
        except Exception:
            # одно неудачное событие не должно останавливать опрос очереди
            log("poll:", traceback.format_exc())
        if not quit_after:
            root.after(100, poll)

    def on_close():
        state["window_size"] = f"{root.winfo_width()}x{root.winfo_height()}"  # запоминаем размер
        save_cfg()  # сначала сохранить (watching=True уйдёт в конфиг)...
        state["watching"] = False  # ...и только потом остановить поток слежения
        try:
            if tray_icon["obj"]:
                tray_icon["obj"].stop()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    # горячие клавиши основного экрана (2.1.1: каждый указательный путь — с клавиатурным)
    def _hot(fn):
        def h(e):
            if isinstance(e.widget, tk.Text):
                return  # в текстовых полях Ctrl+O остаётся штатным
            fn()
        return h
    root.bind_all("<Control-o>", _hot(pick_files))
    root.bind_all("<Control-O>", _hot(pick_folder_files))
    # штатная кнопка сворачивания Windows — прячет окно в трей, а не в панель задач
    def on_minimize(_e):
        if IS_WIN and root.state() == "iconic":
            root.after(30, root.withdraw)
    root.bind("<Unmap>", on_minimize)
    start_tray()
    update_total()
    # запуск свёрнутой в трей, если так настроено
    if cfg.get("start_minimized") and tray_icon["obj"] is not None:
        root.after(400, root.withdraw)
    # возобновить слежение, если оно было включено до перезапуска
    if cfg.get("watching") and Path(v_watch.get().strip()).is_dir():
        root.after(800, toggle_watch)
    poll()
    root.mainloop()

def already_running() -> bool:
    """Защита от второй копии: мьютекс (Windows) или блокировка файла (macOS/Linux).
    На Windows при обнаружении копии выводит её окно наверх."""
    if IS_WIN:
        _ERROR_ALREADY_EXISTS = 183
        ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\TranscriberSingleInstance")
        if ctypes.windll.kernel32.GetLastError() == _ERROR_ALREADY_EXISTS:
            _hwnd = ctypes.windll.user32.FindWindowW(None, "Транскрибатор — аудио/видео в текст")
            if _hwnd:
                ctypes.windll.user32.ShowWindow(_hwnd, 9)          # SW_RESTORE
                ctypes.windll.user32.SetForegroundWindow(_hwnd)
            return True
        return False
    try:
        import fcntl
        _lock = open(BASE / ".single_lock", "wb")
        fcntl.flock(_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return False
    except (ImportError, OSError):
        return True  # не смогли проверить — не мешаем запуску

if __name__ == "__main__":
    enable_dpi_awareness()
    if "--selftest" in sys.argv:
        selftest()
    else:
        try:
            if not already_running():
                run_gui()
        except Exception:
            log(traceback.format_exc())
            raise
