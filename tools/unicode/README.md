# Standard-library table bundle

`tables-16.0.0.tar.gz` holds the generated standard-library modules (Unicode 16.0.0
tables and floating-point conversion tables). It is a checked-in, offline artifact:
builds need no network access, no UCD input files and no Python interpreter.

`crates/compiler/build.rs` verifies the archive and every member against
`manifest.json` (SHA-256), then unpacks them into `$OUT_DIR/stdlib`. The compiler adds
that directory as an extra stdlib search root (`TYPHON_GENERATED_STDLIB`).
Expanded generated sources are not tracked in git.

## Layout

Handwritten algorithms:

- `stdlib/internal/unicode/normalization.py`: Unicode normalization.
- `stdlib/internal/unicode/names.py`: name search, decoding, and algorithmic names.
- `stdlib/internal/numeric/float_dtoa.py`: floating-point conversion, unrelated to Unicode.

Generated modules, relative to the generated stdlib root:

- `internal/unicode/generated/properties.py`: character predicates and case mappings.
- `internal/unicode/generated/single_byte_codecs.py`: single-byte codec tables.
- `internal/unicode/generated/normalization.py`: normalization tables.
- `internal/unicode/generated/metadata.py`: categories, numeric values, and other metadata.
- `internal/unicode/generated/names.py`: names, aliases, and named sequences.
- `internal/numeric/generated/float_dtoa_tables.py`: floating-point conversion tables.

The hot string properties remain separate from metadata, names, and normalization.
Hangul names are algorithmic; CJK unified ideographs use exact assigned ranges
derived from the pinned database. Other names use a shared-word dictionary, and
byte pools use compact LLVM string constants. There is no runtime archive or
dictionary decompression.

`inputs.json` pins the Unicode Character Database files (URLs and SHA-256) the tables
were generated from.

## Updating

Do not edit generated tables by hand. To update them, regenerate the six modules from
the pinned UCD inputs, pack them into a reproducible `tar.gz` (sorted members, zero
mtimes/uids), and update the archive `sha256` plus each file's `sha256`/`size` in
`manifest.json` in the same commit. A stale or modified bundle fails the build.
