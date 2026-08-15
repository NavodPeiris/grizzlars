"""Generate nanobind glue + a .pyi stub for grizzlars_shim.h via litgen.

Usage: generate_bindings.py <shim_header> <output_glue.inc> <output_stub.pyi>

Run automatically as a CMake build step (see CMakeLists.txt) so the glue can
never drift out of sync with the shim header it's generated from.
"""

import sys

import litgen


def main() -> None:
    header_path, glue_path, stub_path = sys.argv[1:4]

    options = litgen.LitgenOptions()
    options.use_nanobind()

    generator = litgen.LitgenGenerator(options)
    generator.process_cpp_file(header_path)

    with open(glue_path, "w", encoding="utf-8") as f:
        f.write(generator.pydef_code())

    with open(stub_path, "w", encoding="utf-8") as f:
        f.write(generator.stub_code())


if __name__ == "__main__":
    main()
