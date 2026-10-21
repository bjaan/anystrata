#!/bin/sh
# vk-glslc.sh - a glslc-shaped CLI shim over whichever SPIR-V front-end the build picked.
#
# CMake drives it as:  cmake -E env VK_GLSLC_MODE=<glslc|glslang> VK_GLSLC_TOOL=<abs path> vk-glslc.sh <f.comp> -o <f.spv> [--target-env=vulkanX.Y] [-O ...]
# The kernels' custom commands keep the glslc-style spelling; the shim translates to
# glslangValidator's ( -V --target-env <env> -O f.comp -o f.spv ) when the vendored
# front-end is in play.
set -e
mode=${VK_GLSLC_MODE:-glslc}
tool=${VK_GLSLC_TOOL:?vk-glslc.sh: VK_GLSLC_TOOL not set}

src=""
out=""
env=vulkan1.0
optimize=0
for a in "$@"; do
    case "$a" in
        -o)        next_is_out=1 ;;
        --target-env=*) env=${a#--target-env=} ;;
        --target-env)   pending_env=1 ;;
        -O|--optimize)  optimize=1 ;;
        *)
            if [ "${next_is_out:-0}" = 1 ]; then out=$a; next_is_out=0
            elif [ "${pending_env:-0}" = 1 ]; then env=$a; pending_env=0
            else src=$a
            fi ;;
    esac
done
[ -n "$src" ] || { echo "vk-glslc.sh: no input shader given" >&2; exit 2; }
[ -n "$out" ] || { echo "vk-glslc.sh: no -o output given" >&2; exit 2; }

if [ "$mode" = glslang ]; then
    # glslangValidator spells the env without '=' and wants -V for Vulkan SPIR-V.
    set -- -V --target-env "$env"
    [ "$optimize" = 1 ] && set -- "$@" -O
    exec "$tool" "$@" "$src" -o "$out"
else
    exec "$tool" --target-env="$env" "$@"
fi
