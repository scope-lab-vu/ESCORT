#!/bin/bash


# & 脚本在任何命令失败时立即退出
set -e


# & Find ale_py roms directory
ALE_ROMS_DIR=$(python -c "import ale_py; print(ale_py.roms.__path__[0])")
echo "ALE ROMs directory: $ALE_ROMS_DIR"

# & Create temp directory
TMP_DIR="/tmp/atari_roms_$$"
mkdir -p "$TMP_DIR"
echo "Downloading ROMs..."

# & Download and extract ROMs from AutoROM source
curl -sL "https://gist.githubusercontent.com/jjshoots/61b22aefce4456920ba99f2c36906eda/raw/00046ac3403768bfe45857610a3d333b8e35e026/Roms.tar.gz.b64" | base64 -d | tar -xzf - -C "$TMP_DIR"

echo "Installing ROMs..."
# & ROMs to install (name matches directory and file in archive)
ROMS=(
    "beam_rider"
    "bowling"
    "ms_pacman"
    "pong"
)

# & Install each ROM
for rom_name in "${ROMS[@]}"; do
    source_file="$TMP_DIR/ROM/${rom_name}/${rom_name}.bin"
    
    if [ -f "$source_file" ]; then
        cp "$source_file" "$ALE_ROMS_DIR/${rom_name}.bin"
        echo "  ✓ $rom_name"
    else
        echo "  ✗ $rom_name (not found)"
    fi
done


# & Cleanup
rm -rf "$TMP_DIR"

echo ""
echo "Done! Installed to: $ALE_ROMS_DIR"
