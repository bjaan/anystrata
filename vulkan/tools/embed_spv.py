#!/usr/bin/env python3
# vulkan/tools/embed_spv.py - turn a .spv into a C array header, the way the tree embeds nothing at run time.
#   embed_spv.py <in.spv> <out.h> <name>  ->  const uint32_t <name>_spv[]; const size_t <name>_spv_len;
import sys

def main():
    src, dst, name = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(src, "rb") as f:
        data = f.read()
    assert len(data) % 4 == 0, "SPIR-V must be a word array"
    words = [int.from_bytes(data[i:i+4], "little") for i in range(0, len(data), 4)]
    with open(dst, "w") as f:
        f.write(f"// generated from {src} by tools/embed_spv.py - do not edit\n")
        f.write("#pragma once\n#include <cstddef>\n#include <cstdint>\n")
        f.write(f"static const uint32_t {name}_spv[] = {{\n")
        for i in range(0, len(words), 8):
            f.write("    " + ",".join(f"0x{w:08x}" for w in words[i:i+8]) + ",\n")
        f.write("};\n")
        f.write(f"static const size_t {name}_spv_len = sizeof({name}_spv);\n")

if __name__ == "__main__":
    main()
