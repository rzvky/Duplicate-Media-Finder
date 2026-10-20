"""File type lists and folder skip rules."""

from __future__ import annotations

# Everyday formats (phones, cameras, web, scans)
STANDARD_EXTS = {
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jfif",
    ".pjpeg",
    ".pjp",
    ".png",
    ".apng",
    ".gif",
    ".bmp",
    ".dib",
    ".tif",
    ".tiff",
    ".webp",
    ".jp2",
    ".j2k",
    ".jpf",
    ".jpx",
    ".ico",
    ".tga",
    ".ppm",
    ".pgm",
    ".pbm",
    ".pnm",
    ".mpo",
}

# Apple (iPhone/iPad/Mac) and newer Android formats
HEIF_EXTS = {".heic", ".heif", ".hif", ".heics", ".heifs"}
AVIF_EXTS = {".avif", ".avifs"}

# Camera RAW formats (Canon, Nikon, Sony, Fuji, Olympus, Panasonic, Pentax,
# Samsung, Leica, Hasselblad, Phase One, GoPro, Adobe DNG incl. iPhone ProRAW)
RAW_EXTS = {
    ".dng",
    ".cr2",
    ".cr3",
    ".crw",
    ".nef",
    ".nrw",
    ".arw",
    ".srf",
    ".sr2",
    ".orf",
    ".rw2",
    ".raf",
    ".pef",
    ".ptx",
    ".srw",
    ".x3f",
    ".3fr",
    ".fff",
    ".erf",
    ".kdc",
    ".dcr",
    ".mrw",
    ".mos",
    ".iiq",
    ".rwl",
    ".raw",
    ".rwz",
    ".mef",
    ".gpr",
    ".bay",
    ".cap",
    ".k25",
}

PHOTO_EXTS = STANDARD_EXTS | HEIF_EXTS | AVIF_EXTS | RAW_EXTS

# Videos (phones, cameras, camcorders, downloads, old PCs)
VIDEO_EXTS = {
    ".mp4",
    ".m4v",
    ".mov",
    ".qt",
    ".3gp",
    ".3g2",
    ".mkv",
    ".webm",
    ".avi",
    ".wmv",
    ".asf",
    ".flv",
    ".f4v",
    ".mpg",
    ".mpeg",
    ".mpe",
    ".m2v",
    ".vob",
    ".ts",
    ".mts",
    ".m2ts",
    ".ogv",
    ".dv",
    ".divx",
    ".rm",
    ".rmvb",
    ".mxf",
}

# Files that travel with a photo (iPhone Live Photo video, Google Takeout Live Photo clip, Apple edit
# sidecar, Lightroom/darktable sidecars, Google Takeout metadata).
LIVE_VIDEO_EXTS = {".mov", ".mp4"}
LIVE_VIDEO_MAX = 25 * 1024 * 1024  # a Live Photo clip is a few MB; bigger videos are separate files
COMPANION_STEM_EXTS = {".mov", ".mp4", ".aae", ".xmp"}
COMPANION_FULLNAME_EXTS = {".xmp", ".json"}
# Files that travel with a video (Apple edit file, sidecars, camera thumbnail, drone/camera subtitles)
VIDEO_SIDECAR_EXTS = {".aae", ".xmp", ".thm", ".srt"}

# Folders never scanned
SKIP_DIR_NAMES = {
    "$recycle.bin",
    "system volume information",
    ".trashes",
    ".trash",
    ".spotlight-v100",
    ".fseventsd",
    "@eadir",
    ".thumbnails",
    ".thumbs",
    "__macosx",
    ".dtrash",
    ".photostructure",
    "node_modules",
}
# Library packages that must never be touched from outside their app
SKIP_DIR_SUFFIXES = {
    ".photoslibrary",
    ".migratedphotolibrary",
    ".photolibrary",
    ".aplibrary",
    ".lrdata",
    ".lrlibrary",
    ".app",
    ".bundle",
}

OUTPUT_DIR_NAME = "_DupeFinder"
REVIEW_BIN_NAME = "Review_Bin"

# Higher = more likely to be the camera original
FORMAT_RANK = {
    "RAW": 5,
    "HEIF": 4,
    "HEIC": 4,
    "JPEG": 3,
    "MPO": 3,
    "TIFF": 3,
    "PNG": 3,
    "JPEG2000": 2,
    "AVIF": 2,
    "WEBP": 2,
    "BMP": 1,
    "GIF": 1,
}


def format_rank(fmt: str, is_raw: bool) -> int:
    if is_raw:
        return FORMAT_RANK["RAW"]
    return FORMAT_RANK.get((fmt or "").upper(), 1)
