import sys
import os
import re
import subprocess
import json
from typing import Optional
import psutil

def is_mirage_browser(pid: int) -> bool:
    mirage_pid_str = os.environ.get('MIRAGE_BROWSER_PID')
    if not mirage_pid_str or not pid: return False
    try:
        mirage_pid = int(mirage_pid_str)
        if pid == mirage_pid: return True
        try:
            parent = psutil.Process(mirage_pid)
            return pid in [c.pid for c in parent.children(recursive=True)]
        except psutil.NoSuchProcess:
            pass
    except ValueError:
        pass
    return False

# Cache for the session type detection
_session_type: Optional[str] = None

# Known browser WM_CLASS values (more reliable than title matching)
BROWSER_WM_CLASSES = {
    "google-chrome", "google-chrome-stable", "google-chrome-beta", "google-chrome-unstable",
    "chromium", "chromium-browser",
    "firefox", "firefox-esr", "Navigator",  # Navigator is Firefox's internal class
    "brave", "brave-browser",
    "microsoft-edge", "msedge",
    "safari",
    "opera",
    "vivaldi", "vivaldi-stable",
    "arc",  # Arc browser class
    "epiphany", "org.gnome.Epiphany",  # GNOME Web
    "midori",
    "falkon",
    "konqueror",
    "webkit2gtk",  # Generic WebKit
}

# Browser patterns for title matching (use word boundaries)
BROWSER_TITLE_PATTERNS = [
    r'\bgoogle\s+chrome\b',
    r'\bchromium\b',
    r'\bfirefox\b',
    r'\bmozilla\s+firefox\b',
    r'\bbrave\b',
    r'\bmicrosoft\s+edge\b',
    r'\bsafari\b',
    r'\bopera\b',
    r'\bvivaldi\b',
]

_browser_title_re = re.compile('|'.join(BROWSER_TITLE_PATTERNS), re.IGNORECASE)

def _get_session_type() -> str:
    """Detect if running on X11 or Wayland."""
    global _session_type
    if _session_type is not None:
        return _session_type

    if os.environ.get("WAYLAND_DISPLAY"):
        _session_type = "wayland"
    elif os.environ.get("XDG_SESSION_TYPE") == "wayland":
        _session_type = "wayland"
    else:
        _session_type = "x11"
    return _session_type

def _get_active_window_wayland() -> tuple[str, str, int]:
    """Get active window on Wayland compositors."""
    # Try GNOME Shell via gdbus
    try:
        result = subprocess.run(
            ['gdbus', 'call', '--session', '--dest', 'org.gnome.Shell',
             '--object-path', '/org/gnome/Shell', '--method',
             'org.gnome.Shell.Eval',
             'global.display.focus_window ? global.display.focus_window.get_wm_class() + "|" + global.display.focus_window.get_title() : ""'],
            capture_output=True, text=True, timeout=1
        )
        if result.returncode == 0 and '|' in result.stdout:
            output = result.stdout.strip()
            if "'" in output:
                parts = output.split("'")
                if len(parts) >= 4:
                    value = parts[3]
                    if '|' in value:
                        wm_class, title = value.split('|', 1)
                        return title, wm_class, 0
    except Exception:
        pass

    # Try Hyprland
    try:
        result = subprocess.run(['hyprctl', 'activewindow', '-j'], capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data.get('title', ''), data.get('class', ''), data.get('pid', 0)
    except Exception:
        pass

    # Try Sway/i3
    try:
        result = subprocess.run(['swaymsg', '-t', 'get_tree'], capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            tree = json.loads(result.stdout)
            focused = _find_focused_sway(tree)
            if focused:
                return focused.get('name', ''), focused.get('app_id', '') or focused.get('window_properties', {}).get('class', ''), focused.get('pid', 0)
    except Exception:
        pass

    return "", "", 0

def _find_focused_sway(node: dict) -> Optional[dict]:
    if node.get('focused'):
        return node
    for child in node.get('nodes', []) + node.get('floating_nodes', []):
        result = _find_focused_sway(child)
        if result:
            return result
    return None

def _get_active_window_x11() -> tuple[str, str, int]:
    """Get active window on X11."""
    try:
        root_proc = subprocess.run(
            ['xprop', '-root', '_NET_ACTIVE_WINDOW'],
            capture_output=True, text=True, timeout=0.5
        )
        if root_proc.returncode != 0:
            return "", "", 0
            
        root = root_proc.stdout.strip()
        window_id = root.split()[-1]

        if window_id == '0x0':
            return "", "", 0

        title = ""
        wm_class = ""
        pid = 0

        try:
            title_proc = subprocess.run(
                ['xprop', '-id', window_id, 'WM_NAME'],
                capture_output=True, text=True, timeout=0.5
            )
            if title_proc.returncode == 0 and '"' in title_proc.stdout:
                title = title_proc.stdout.strip().split('"')[1]
        except Exception:
            pass

        try:
            class_proc = subprocess.run(
                ['xprop', '-id', window_id, 'WM_CLASS'],
                capture_output=True, text=True, timeout=0.5
            )
            if class_proc.returncode == 0 and '"' in class_proc.stdout:
                wm_class = class_proc.stdout.strip().split('"')[1]
        except Exception:
            pass
            
        try:
            pid_proc = subprocess.run(
                ['xprop', '-id', window_id, '_NET_WM_PID'],
                capture_output=True, text=True, timeout=0.5
            )
            if pid_proc.returncode == 0 and '=' in pid_proc.stdout:
                pid_str = pid_proc.stdout.strip().split('=')[-1].strip()
                if pid_str.isdigit():
                    pid = int(pid_str)
        except Exception:
            pass

        return title, wm_class, pid
    except Exception:
        return "", "", 0

def get_active_window_title() -> tuple[str, str, int]:
    """Returns (title, class/app_name, pid) of the currently active window, cross-platform."""
    try:
        if sys.platform == "win32":
            import ctypes
            import win32process
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            return buf.value if buf.value else "", "", pid

        elif sys.platform == "darwin":
            script = 'tell application "System Events" to get {name, unix id} of first application process whose frontmost is true'
            proc = subprocess.run(['osascript', '-e', script], capture_output=True, text=True, timeout=0.5)
            output = proc.stdout.strip().split(", ")
            app_name = output[0] if len(output) > 0 else ""
            pid = int(output[1]) if len(output) > 1 and output[1].isdigit() else 0

            script_title = f'tell application "System Events" to tell process "{app_name}" to get name of front window'
            proc_title = subprocess.run(['osascript', '-e', script_title], capture_output=True, text=True, timeout=0.5)
            title = proc_title.stdout.strip()
            return title if title else app_name, app_name, pid

        elif sys.platform.startswith("linux"):
            session_type = _get_session_type()
            if session_type == "wayland":
                return _get_active_window_wayland()
            else:
                return _get_active_window_x11()

    except Exception:
        pass

    return "", "", 0

def is_browser(title: str, app_class: str, pid: int = 0) -> bool:
    if pid > 0 and is_mirage_browser(pid):
        return True
    if os.environ.get("MIRAGE_BROWSER_PID"):
        return False

    title_lower = title.lower() if title else ""
    class_lower = app_class.lower() if app_class else ""

    non_browsers = [
        "gnome-terminal", "konsole", "xterm", "iterm", "alacritty", "kitty",
        "terminal", "tilix", "terminator", "urxvt", "st",
        "code", "vscode", "vscodium",
        "sublime", "atom", "gedit", "kate", "notepad", "text editor",
        "nautilus", "dolphin", "thunar", "nemo",
        "libreoffice", "openoffice",
        "gimp", "inkscape", "krita",
        "vlc", "mpv", "totem",
        "slack", "discord", "telegram", "signal",
        "thunderbird", "evolution", "geary",
        "claude",
    ]
    combined = f"{title_lower} {class_lower}"
    for nb in non_browsers:
        if re.search(rf"\b{re.escape(nb)}\b", combined):
            return False

    if class_lower:
        for browser_class in BROWSER_WM_CLASSES:
            if browser_class in class_lower:
                return True

    if title_lower and _browser_title_re.search(title_lower):
        return True
    if class_lower and _browser_title_re.search(class_lower):
        return True

    return False
