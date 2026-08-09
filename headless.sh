pkill -9 xpra || true
pkill -9 Xvfb || true

export DISPLAY=:99
Xvfb :99 -screen 0 640x480x24 -nolisten tcp &
export DISPLAY=:99

export LIBGL_ALWAYS_SOFTWARE=1
export PYOPENGL_PLATFORM=osmesa

source .venv/bin/activate
python interactive.py