#!/usr/bin/env bash
set -euo pipefail

arch="${1:?usage: build_windows.sh <arch>}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/windows-${arch}"
PACKAGE_NAME="nanotts-windows-${arch}"
PACKAGE_DIR="${BUILD_DIR}/${PACKAGE_NAME}"
DIST_DIR="${ROOT_DIR}/dist"
MINGW_PREFIX="${MINGW_PREFIX:-/mingw64}"

split_flags() {
    local output
    output="$("$@")"
    if [[ -z "${output}" ]]; then
        return 0
    fi

    # shellcheck disable=SC2206
    local parts=(${output})
    printf '%s\n' "${parts[@]}"
}

copy_common_files() {
    cp "${ROOT_DIR}/README.md" "${PACKAGE_DIR}/"
    cp "${ROOT_DIR}/nanotts.h" "${PACKAGE_DIR}/"
    cp "${ROOT_DIR}/example.py" "${PACKAGE_DIR}/"
}

copy_windows_deps() {
    local dest_dir="${1:?missing destination}"
    local seed="${2:?missing seed binary}"
    declare -A seen=()
    local queue=("${seed}")

    while ((${#queue[@]})); do
        local current="${queue[0]}"
        queue=("${queue[@]:1}")

        while IFS= read -r dep; do
            [[ -n "${dep}" ]] || continue
            [[ "${dep}" == "${MINGW_PREFIX}/bin/"* ]] || continue

            if [[ -n "${seen[${dep}]:-}" ]]; then
                continue
            fi
            seen["${dep}"]=1

            local dest_path="${dest_dir}/$(basename "${dep}")"
            cp -L "${dep}" "${dest_path}"
            chmod 0644 "${dest_path}"
            queue+=("${dest_path}")
        done < <(ldd "${current}" | awk '/=>/ {print $3} /^\/mingw64\// {print $1}')
    done
}

if [[ ! -f "${ROOT_DIR}/model.bin" ]]; then
    echo "model.bin not found in repository root" >&2
    exit 1
fi

rm -rf "${PACKAGE_DIR}"
mkdir -p "${BUILD_DIR}" "${PACKAGE_DIR}" "${DIST_DIR}"
rm -f "${BUILD_DIR}"/*.o "${BUILD_DIR}"/embed_weights.o "${BUILD_DIR}"/_embed_w.s

CC_BIN="${CC:-gcc}"
CXX_BIN="${CXX:-g++}"
TARGET_NAME="nanotts.dll"
EMBED_ASM="${BUILD_DIR}/_embed_w.s"
EMBED_OBJ="${BUILD_DIR}/embed_weights.o"

COMMON_CFLAGS=(-O2 -Wall -Wextra -std=c11 -DUSE_OPENBLAS -DEMBED_WEIGHTS -fPIC)
COMMON_CXXFLAGS=(-O2 -Wall -Wextra -std=c++17 -DEMBED_WEIGHTS -fPIC)
mapfile -t INCLUDES < <(split_flags pkg-config --cflags openblas sentencepiece)
mapfile -t PKG_LIBS < <(split_flags pkg-config --libs openblas sentencepiece)
LDFLAGS=(
    -shared
    "-Wl,--out-implib,${BUILD_DIR}/libnanotts.dll.a"
    -static-libgcc
    -static-libstdc++
    -lm
    "${PKG_LIBS[@]}"
)

cat > "${EMBED_ASM}" <<'EOF'
.global weights_data
.global weights_size
.section .rdata,"dr"
.balign 8
weights_data:
.incbin "model.bin"
weights_size:
.quad . - weights_data
EOF

C_SRCS=(nanotts.c tensor.c ops.c tts.c codec.c prompt.c audio.c)
CXX_SRCS=(sentencepiece.cpp)
OBJECTS=()

for src in "${C_SRCS[@]}"; do
    obj="${BUILD_DIR}/$(basename "${src%.c}.o")"
    "${CC_BIN}" "${COMMON_CFLAGS[@]}" "${INCLUDES[@]}" -c "${ROOT_DIR}/${src}" -o "${obj}"
    OBJECTS+=("${obj}")
done

for src in "${CXX_SRCS[@]}"; do
    obj="${BUILD_DIR}/$(basename "${src%.cpp}.o")"
    "${CXX_BIN}" "${COMMON_CXXFLAGS[@]}" "${INCLUDES[@]}" -c "${ROOT_DIR}/${src}" -o "${obj}"
    OBJECTS+=("${obj}")
done

(
    cd "${ROOT_DIR}"
    as -o "${EMBED_OBJ}" "${EMBED_ASM}"
)

"${CXX_BIN}" -o "${PACKAGE_DIR}/${TARGET_NAME}" "${OBJECTS[@]}" "${EMBED_OBJ}" "${LDFLAGS[@]}"
cp "${PACKAGE_DIR}/${TARGET_NAME}" "${PACKAGE_DIR}/moss_tts.dll"

copy_common_files
copy_windows_deps "${PACKAGE_DIR}" "${PACKAGE_DIR}/${TARGET_NAME}"

(
    cd "${BUILD_DIR}"
    rm -f "${DIST_DIR}/${PACKAGE_NAME}.zip"
    zip -qr "${DIST_DIR}/${PACKAGE_NAME}.zip" "${PACKAGE_NAME}"
)

echo "created ${DIST_DIR}/${PACKAGE_NAME}.zip"
