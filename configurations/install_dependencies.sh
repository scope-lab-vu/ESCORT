#!/bin/bash


echo "Installing system dependencies..."

# Add deadsnakes PPA for Python 3.14
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
# build-essential:  C/C++ compiler toolchain (gcc, g++, make)
# cmake:            Build system for compiling C/C++ projects (e.g. ale-py)
# ffmpeg:           Video encoding/decoding (used by imageio for MP4 output)
# libsdl2-dev:      SDL2 headers for Atari game rendering
# libopencv-dev:    OpenCV headers for image processing
# python3.14:       Python 3.14 interpreter
# python3.14-dev:   Python 3.14 C headers (needed to build C extensions)
# python3.14-venv:  venv module for Python 3.14
sudo apt install -y \
    build-essential \
    cmake \
    ffmpeg \
    libsdl2-dev \
    libopencv-dev \
    python3.14 \
    python3.14-dev \
    python3.14-venv
sudo apt update


echo "Done! All dependencies installed."
