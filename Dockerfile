FROM espressif/idf:release-v5.5

# X11 runtime libs required by PyQt5 for the live CSI viewer GUI
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    libxcb-cursor0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb1 \
    libx11-xcb1 \
    libsm6 \
    libice6 \
    libdbus-1-3 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Install python deps for the CSI viewer tool into IDF's own python venv
RUN IDF_PATH_FORCE=1 . "$IDF_PATH/export.sh" && pip install --no-cache-dir \
    pyserial \
    pandas \
    numpy \
    PyQt5 \
    pyqtgraph \
    matplotlib \
    scipy \
    statsmodels

WORKDIR /workspace
