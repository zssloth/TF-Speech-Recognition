package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//WaveNet-ASR/...",
    ],
)

filegroup(
    name = "py_srcs",
    data = glob([
        "**/*.py",
    ]),
)

py_library(
    name = "asr_model",
    srcs = ["asr_model.py"],
)

py_binary(
    name = "asr_main",
    srcs = [
        "asr_main.py",
    ],
    deps = [
        ":data",
        ":asr_model",
    ],
)

py_library(
    name = "data",
    srcs = ["data.py"],
    deps = [
        ":preprocess_data",
    ],
)

py_library(
    name = "preprocess_data",
    srcs = ["preprocess_data.py"],
)

