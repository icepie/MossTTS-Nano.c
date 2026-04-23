#!/usr/bin/env bash
set -euo pipefail

platform="${1:?usage: build_unix.sh <linux|macos> <arch>}"
arch="${2:?usage: build_unix.sh <linux|macos> <arch>}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/${platform}-${arch}"
PACKAGE_NAME="nanotts-${platform}-${arch}"
PACKAGE_DIR="${BUILD_DIR}/${PACKAGE_NAME}"
DIST_DIR="${ROOT_DIR}/dist"

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

copy_linux_deps() {
    local dest_dir="${1:?missing destination}"
    local seed="${2:?missing seed binary}"
    declare -A seen=()
    local queue=("${seed}")

    while ((${#queue[@]})); do
        local current="${queue[0]}"
        queue=("${queue[@]:1}")

        while IFS= read -r dep; do
            [[ -n "${dep}" ]] || continue

            case "$(basename "${dep}")" in
                linux-vdso.so.*|libc.so.*|libm.so.*|libpthread.so.*|libdl.so.*|librt.so.*|libresolv.so.*|ld-linux*.so.*)
                    continue
                    ;;
            esac

            if [[ -n "${seen[${dep}]:-}" ]]; then
                continue
            fi
            seen["${dep}"]=1

            local dest_path="${dest_dir}/$(basename "${dep}")"
            cp -L "${dep}" "${dest_path}"
            chmod 0644 "${dest_path}"
            queue+=("${dest_path}")
        done < <(ldd "${current}" | awk '/=> \// {print $3} /^\/[^ ]+/ {print $1}')
    done
}

copy_macos_deps() {
    local dest_dir="${1:?missing destination}"
    local seed="${2:?missing seed binary}"
    declare -A seen=()
    local queue=("${seed}")

    while ((${#queue[@]})); do
        local current="${queue[0]}"
        queue=("${queue[@]:1}")

        while IFS= read -r dep; do
            [[ -n "${dep}" ]] || continue

            if [[ -n "${seen[${dep}]:-}" ]]; then
                continue
            fi
            seen["${dep}"]=1

            local dest_path="${dest_dir}/$(basename "${dep}")"
            cp -L "${dep}" "${dest_path}"
            chmod u+w "${dest_path}"
            queue+=("${dest_path}")
        done < <(otool -L "${current}" | awk 'NR > 1 {print $1}' | grep -vE '^/System/Library/|^/usr/lib/' || true)
    done

    while IFS= read -r file; do
        [[ -f "${file}" ]] || continue
        chmod u+w "${file}"

        while IFS= read -r dep; do
            [[ -n "${dep}" ]] || continue
            install_name_tool -change "${dep}" "@loader_path/$(basename "${dep}")" "${file}"
        done < <(otool -L "${file}" | awk 'NR > 1 {print $1}' | grep -vE '^/System/Library/|^/usr/lib/' || true)

        if [[ "${file}" == *.dylib ]]; then
            install_name_tool -id "@loader_path/$(basename "${file}")" "${file}" || true
        fi

        codesign --force --sign - "${file}" >/dev/null 2>&1 || true
    done < <(find "${PACKAGE_DIR}" -maxdepth 1 -type f \( -name '*.dylib' -o -perm -111 \))
}

if [[ ! -f "${ROOT_DIR}/model.bin" ]]; then
    echo "model.bin not found in repository root" >&2
    exit 1
fi

rm -rf "${PACKAGE_DIR}"
mkdir -p "${BUILD_DIR}" "${PACKAGE_DIR}" "${DIST_DIR}"
rm -f "${BUILD_DIR}"/*.o "${BUILD_DIR}"/embed_weights.o "${BUILD_DIR}"/_embed_w.s

CC_BIN="${CC:-cc}"
CXX_BIN="${CXX:-c++}"
COMMON_CFLAGS=(-O2 -Wall -Wextra -std=c11 -DUSE_OPENBLAS -DEMBED_WEIGHTS)
COMMON_CXXFLAGS=(-O2 -Wall -Wextra -std=c++17 -DEMBED_WEIGHTS)
INCLUDES=()
LDFLAGS=()
TARGET_NAME=""
EMBED_ASM="${BUILD_DIR}/_embed_w.s"
EMBED_OBJ="${BUILD_DIR}/embed_weights.o"

case "${platform}" in
    linux)
        TARGET_NAME="libnanotts.so"
        COMMON_CFLAGS+=(-D_POSIX_C_SOURCE=200809L -fPIC)
        COMMON_CXXFLAGS+=(-fPIC)
        mapfile -t INCLUDES < <(split_flags pkg-config --cflags openblas sentencepiece)
        mapfile -t LDFLAGS < <(split_flags pkg-config --libs openblas sentencepiece)
        LDFLAGS=(
            -shared
            -Wl,--disable-new-dtags
            "-Wl,-rpath,\$ORIGIN"
            -lm
            -pthread
            "${LDFLAGS[@]}"
        )
        cat > "${EMBED_ASM}" <<'EOF'
.global weights_data
.global weights_size
.section .rodata
.balign 8
weights_data:
.incbin "model.bin"
weights_size:
.quad . - weights_data
EOF
        ;;
    macos)
        TARGET_NAME="libnanotts.dylib"
        OPENBLAS_PREFIX="${OPENBLAS_PREFIX:-$(brew --prefix openblas)}"
        SENTENCEPIECE_PREFIX="${SENTENCEPIECE_PREFIX:-$(brew --prefix sentencepiece)}"
        INCLUDES=(
            "-I${OPENBLAS_PREFIX}/include"
            "-I${SENTENCEPIECE_PREFIX}/include"
        )
        LDFLAGS=(
            -dynamiclib
            -lm
            "-L${OPENBLAS_PREFIX}/lib"
            "-L${SENTENCEPIECE_PREFIX}/lib"
            -lopenblas
            -lsentencepiece
            -Wl,-rpath,@loader_path
        )
        cat > "${EMBED_ASM}" <<'EOF'
.global _weights_data
.global _weights_size
.section __DATA,__weights
.align 4
_weights_data:
.incbin "model.bin"
_weights_size:
.quad . - _weights_data
EOF
        ;;
    *)
        echo "unsupported platform: ${platform}" >&2
        exit 1
        ;;
esac

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

copy_common_files

if [[ "${platform}" == "linux" ]]; then
    copy_linux_deps "${PACKAGE_DIR}" "${PACKAGE_DIR}/${TARGET_NAME}"
else
    copy_macos_deps "${PACKAGE_DIR}" "${PACKAGE_DIR}/${TARGET_NAME}"
fi

(
    cd "${BUILD_DIR}"
    rm -f "${DIST_DIR}/${PACKAGE_NAME}.zip"
    zip -qr "${DIST_DIR}/${PACKAGE_NAME}.zip" "${PACKAGE_NAME}"
)

echo "created ${DIST_DIR}/${PACKAGE_NAME}.zip"
