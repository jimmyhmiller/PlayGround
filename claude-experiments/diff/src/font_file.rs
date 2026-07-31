//! Reading a font FILE for the five numbers `next/font/local` needs.
//!
//! `next/font/local`'s whole value over a hand-written `@font-face` is the
//! metric-matched fallback: a `local("Arial")` face carrying `size-adjust` /
//! `ascent-override` / `descent-override` / `line-gap-override` scaled so the
//! substituted system font occupies the same space as the real one, and the page
//! therefore does not reflow when the webfont finishes loading. Next computes those
//! four numbers by running fontkit over the font binary
//! (`@next/font/dist/local/get-fallback-metrics-from-font-file.js`); there is no table
//! to look them up in, because the font is the app's own.
//!
//! So this module is a small, exact font reader: enough of the SFNT container (plus
//! the WOFF and WOFF2 wrappers around it) to reach
//!
//! * `head.unitsPerEm`,
//! * `hhea.ascender` / `descender` / `lineGap` — fontkit's `font.ascent` is
//!   `this.hhea.ascent`, *not* the OS/2 typographic metrics, so this reads hhea and
//!   only hhea,
//! * and the advance width of each glyph in Next's fixed sample string, via `cmap` +
//!   `hmtx`, which is how `calcAverageWidth` derives `azAvgWidth`.
//!
//! Everything it cannot handle is a hard error naming the file and the exact reason
//! (an unknown `cmap` format, a truncated table, a WOFF2 whose Brotli stream will not
//! decompress). A font whose metrics cannot be read must never silently become
//! `size-adjust: 100%` — that is precisely the layout shift the feature exists to
//! prevent, and it would be invisible.

use std::path::Path;

/// The metrics fontkit exposes to `getFallbackMetricsFromFontFile`, in font units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FontMetrics {
    /// `hhea.ascender`.
    pub ascent: f64,
    /// `hhea.descender` (negative in every real font).
    pub descent: f64,
    /// `hhea.lineGap`.
    pub line_gap: f64,
    /// `head.unitsPerEm`.
    pub units_per_em: f64,
    /// Next's `calcAverageWidth`: the mean advance width over its fixed sample string,
    /// or `None` when the font lacks a glyph for one of those characters — in which
    /// case Next's `sizeAdjust` is 1 (see `getFallbackMetricsFromFontFile`).
    pub az_avg_width: Option<f64>,
}

/// Next's sample string (`@next/font/dist/local/get-fallback-metrics-from-font-file.js`,
/// `calcAverageWidth`): letter-frequency weighted, with six spaces standing in for word
/// breaks. The average is taken over these 43 characters, so it must match byte for
/// byte or every `size-adjust` diffpack emits is wrong.
const AVG_CHARACTERS: &str = "aaabcdeeeefghiijklmnnoopqrrssttuvwxyz      ";

/// Read the metrics of the font at `path`. Supports the container formats
/// `next/font/local` accepts (`.ttf`, `.otf`, `.woff`, `.woff2`).
pub fn read_metrics(path: &Path) -> Result<FontMetrics, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("cannot read font file {}: {error}", path.display()))?;
    read_metrics_from_bytes(&bytes, path)
}

/// The same, from bytes already in hand (the build reads each font once, for its
/// content hash and its metrics together).
pub fn read_metrics_from_bytes(bytes: &[u8], path: &Path) -> Result<FontMetrics, String> {
    let tables = Tables::parse(bytes, path)?;
    let head = tables.get(b"head", path)?;
    let hhea = tables.get(b"hhea", path)?;
    let units_per_em = f64::from(be_u16(head, 18, b"head", path)?);
    if units_per_em == 0.0 {
        return Err(format!("{}: head.unitsPerEm is 0", path.display()));
    }
    let ascent = f64::from(be_i16(hhea, 4, b"hhea", path)?);
    let descent = f64::from(be_i16(hhea, 6, b"hhea", path)?);
    let line_gap = f64::from(be_i16(hhea, 8, b"hhea", path)?);
    let az_avg_width = average_width(&tables, path)?;
    Ok(FontMetrics { ascent, descent, line_gap, units_per_em, az_avg_width })
}

/// `calcAverageWidth`: look every character of [`AVG_CHARACTERS`] up in `cmap`, and if
/// the font has a glyph for all of them, average their `hmtx` advance widths. A single
/// missing glyph makes the whole average `None`, exactly as fontkit's `hasAllChars`
/// check does.
fn average_width(tables: &Tables<'_>, path: &Path) -> Result<Option<f64>, String> {
    let cmap = tables.get(b"cmap", path)?;
    let lookup = CmapLookup::parse(cmap, path)?;
    let maxp = tables.get(b"maxp", path)?;
    let num_glyphs = be_u16(maxp, 4, b"maxp", path)?;
    let hhea = tables.get(b"hhea", path)?;
    let num_h_metrics = be_u16(hhea, 34, b"hhea", path)?;
    let hmtx = tables.get(b"hmtx", path)?;
    let advances = HorizontalMetrics { hmtx, num_h_metrics, num_glyphs, transformed: tables.hmtx_transformed };

    let mut total = 0f64;
    let mut count = 0usize;
    for character in AVG_CHARACTERS.chars() {
        let glyph = lookup.glyph(character as u32);
        if glyph == 0 {
            return Ok(None);
        }
        total += f64::from(advances.advance(glyph, path)?);
        count += 1;
    }
    if count == 0 {
        return Ok(None);
    }
    Ok(Some(total / count as f64))
}

/// The `hmtx` advance-width array. Glyphs past `numberOfHMetrics` all share the last
/// entry's advance (the OpenType rule for monospaced tails).
struct HorizontalMetrics<'a> {
    hmtx: &'a [u8],
    num_h_metrics: u16,
    num_glyphs: u16,
    /// A WOFF2 `hmtx` may be stored in the transformed form, whose advance-width array
    /// is the same `numberOfHMetrics` big-endian u16s but with no interleaved bearings.
    transformed: bool,
}

impl HorizontalMetrics<'_> {
    fn advance(&self, glyph: u16, path: &Path) -> Result<u16, String> {
        if self.num_h_metrics == 0 {
            return Err(format!("{}: hhea.numberOfHMetrics is 0", path.display()));
        }
        if glyph >= self.num_glyphs {
            return Err(format!(
                "{}: glyph id {glyph} is past maxp.numGlyphs ({})",
                path.display(),
                self.num_glyphs
            ));
        }
        let index = glyph.min(self.num_h_metrics - 1);
        // Transformed hmtx: `flags` byte, then advanceWidth[numberOfHMetrics].
        // Untransformed: longHorMetric[numberOfHMetrics] = (advanceWidth, lsb).
        let offset = if self.transformed {
            1 + usize::from(index) * 2
        } else {
            usize::from(index) * 4
        };
        be_u16(self.hmtx, offset, b"hmtx", path)
    }
}

/// A parsed table directory: the tag -> bytes map, whatever container it came in.
struct Tables<'a> {
    entries: Vec<([u8; 4], std::borrow::Cow<'a, [u8]>)>,
    hmtx_transformed: bool,
}

impl<'a> Tables<'a> {
    fn get(&self, tag: &[u8; 4], path: &Path) -> Result<&[u8], String> {
        self.entries
            .iter()
            .find(|(entry, _)| entry == tag)
            .map(|(_, data)| data.as_ref())
            .ok_or_else(|| {
                format!(
                    "{}: the font has no `{}` table, so its fallback metrics cannot be \
                     derived (next/font/local computes size-adjust from head/hhea/hmtx/cmap)",
                    path.display(),
                    String::from_utf8_lossy(tag),
                )
            })
    }

    fn parse(bytes: &'a [u8], path: &Path) -> Result<Self, String> {
        if bytes.len() < 4 {
            return Err(format!("{}: font file is {} bytes", path.display(), bytes.len()));
        }
        match &bytes[0..4] {
            b"wOF2" => Self::parse_woff2(bytes, path),
            b"wOFF" => Self::parse_woff(bytes, path),
            b"ttcf" => Err(format!(
                "{}: TrueType Collections (ttcf) are not a format `next/font/local` \
                 accepts (only .ttf/.otf/.woff/.woff2)",
                path.display()
            )),
            _ => Self::parse_sfnt(bytes, path),
        }
    }

    /// A bare SFNT (`.ttf`/`.otf`): 12-byte header then 16-byte directory entries.
    fn parse_sfnt(bytes: &'a [u8], path: &Path) -> Result<Self, String> {
        let num_tables = be_u16(bytes, 4, b"sfnt", path)?;
        let mut entries = Vec::with_capacity(usize::from(num_tables));
        for index in 0..usize::from(num_tables) {
            let base = 12 + index * 16;
            let tag = tag_at(bytes, base, path)?;
            let offset = be_u32(bytes, base + 8, b"sfnt", path)? as usize;
            let length = be_u32(bytes, base + 12, b"sfnt", path)? as usize;
            let end = offset.checked_add(length).ok_or_else(|| {
                format!("{}: table `{}` overflows the file", path.display(), tag_name(&tag))
            })?;
            // A table declared past EOF is a corrupt file, not something to skip: the
            // metrics would silently come out wrong.
            let data = bytes.get(offset..end.min(bytes.len())).ok_or_else(|| {
                format!(
                    "{}: table `{}` claims bytes {offset}..{end} of a {}-byte file",
                    path.display(),
                    tag_name(&tag),
                    bytes.len()
                )
            })?;
            entries.push((tag, std::borrow::Cow::Borrowed(data)));
        }
        Ok(Self { entries, hmtx_transformed: false })
    }

    /// WOFF (v1): a 44-byte header, 20-byte directory entries, and each table
    /// individually zlib-compressed when `compLength < origLength`.
    fn parse_woff(bytes: &'a [u8], path: &Path) -> Result<Self, String> {
        let num_tables = be_u16(bytes, 12, b"woff", path)?;
        let mut entries = Vec::with_capacity(usize::from(num_tables));
        for index in 0..usize::from(num_tables) {
            let base = 44 + index * 20;
            let tag = tag_at(bytes, base, path)?;
            let offset = be_u32(bytes, base + 4, b"woff", path)? as usize;
            let comp_length = be_u32(bytes, base + 8, b"woff", path)? as usize;
            let orig_length = be_u32(bytes, base + 12, b"woff", path)? as usize;
            let end = offset.checked_add(comp_length).ok_or_else(|| {
                format!("{}: woff table `{}` overflows the file", path.display(), tag_name(&tag))
            })?;
            let raw = bytes.get(offset..end).ok_or_else(|| {
                format!(
                    "{}: woff table `{}` claims bytes {offset}..{end} of a {}-byte file",
                    path.display(),
                    tag_name(&tag),
                    bytes.len()
                )
            })?;
            let data = if comp_length < orig_length {
                use std::io::Read;
                let mut out = Vec::with_capacity(orig_length);
                flate2::read::ZlibDecoder::new(raw).read_to_end(&mut out).map_err(|error| {
                    format!(
                        "{}: woff table `{}` will not zlib-decompress: {error}",
                        path.display(),
                        tag_name(&tag)
                    )
                })?;
                std::borrow::Cow::Owned(out)
            } else {
                std::borrow::Cow::Borrowed(raw)
            };
            entries.push((tag, data));
        }
        Ok(Self { entries, hmtx_transformed: false })
    }

    /// WOFF2: a 48-byte header, a variable-length directory, then ONE Brotli stream
    /// holding every table back to back in directory order.
    fn parse_woff2(bytes: &'a [u8], path: &Path) -> Result<Self, String> {
        let num_tables = usize::from(be_u16(bytes, 12, b"wof2", path)?);
        let total_compressed = be_u32(bytes, 20, b"wof2", path)? as usize;
        let mut cursor = 48usize;
        // (tag, length inside the decompressed stream, transformed)
        let mut directory: Vec<([u8; 4], usize, bool)> = Vec::with_capacity(num_tables);
        for _ in 0..num_tables {
            let flags = *bytes.get(cursor).ok_or_else(|| truncated(path, "woff2 table directory"))?;
            cursor += 1;
            let known = flags & 0x3f;
            let tag = if known == 63 {
                let tag = tag_at(bytes, cursor, path)?;
                cursor += 4;
                tag
            } else {
                *KNOWN_TABLE_TAGS.get(usize::from(known)).ok_or_else(|| {
                    format!("{}: woff2 known-table index {known} is out of range", path.display())
                })?
            };
            let orig_length = read_base128(bytes, &mut cursor, path)?;
            // The spec: `glyf`/`loca` are transformed when the version is 0 (3 is the
            // null transform); every other table is transformed when it is NOT 0.
            let version = flags >> 6;
            let transformed = if &tag == b"glyf" || &tag == b"loca" {
                version == 0
            } else {
                version != 0
            };
            let length =
                if transformed { read_base128(bytes, &mut cursor, path)? } else { orig_length };
            directory.push((tag, length, transformed));
        }
        let compressed = bytes
            .get(cursor..cursor + total_compressed)
            .ok_or_else(|| truncated(path, "woff2 compressed stream"))?;
        let mut decompressed = Vec::new();
        brotli_decompressor::BrotliDecompress(&mut &compressed[..], &mut decompressed).map_err(
            |error| {
                format!("{}: the woff2 Brotli stream will not decompress: {error}", path.display())
            },
        )?;

        let mut entries = Vec::with_capacity(num_tables);
        let mut hmtx_transformed = false;
        let mut offset = 0usize;
        for (tag, length, transformed) in directory {
            let end = offset.checked_add(length).ok_or_else(|| {
                format!("{}: woff2 table `{}` overflows the stream", path.display(), tag_name(&tag))
            })?;
            let data = decompressed.get(offset..end).ok_or_else(|| {
                format!(
                    "{}: woff2 table `{}` claims bytes {offset}..{end} of a {}-byte \
                     decompressed stream",
                    path.display(),
                    tag_name(&tag),
                    decompressed.len()
                )
            })?;
            if &tag == b"hmtx" && transformed {
                hmtx_transformed = true;
            }
            entries.push((tag, std::borrow::Cow::Owned(data.to_vec())));
            offset = end;
        }
        Ok(Self { entries, hmtx_transformed })
    }
}

/// WOFF2's `UIntBase128`: up to five 7-bit groups, most significant first, with the
/// high bit marking continuation.
fn read_base128(bytes: &[u8], cursor: &mut usize, path: &Path) -> Result<usize, String> {
    let mut value: u32 = 0;
    for index in 0..5 {
        let byte = *bytes.get(*cursor).ok_or_else(|| truncated(path, "woff2 UIntBase128"))?;
        *cursor += 1;
        if index == 0 && byte == 0x80 {
            return Err(format!("{}: woff2 UIntBase128 has a leading zero", path.display()));
        }
        value = value
            .checked_mul(128)
            .and_then(|v| v.checked_add(u32::from(byte & 0x7f)))
            .ok_or_else(|| format!("{}: woff2 UIntBase128 overflows 32 bits", path.display()))?;
        if byte & 0x80 == 0 {
            return Ok(value as usize);
        }
    }
    Err(format!("{}: woff2 UIntBase128 is longer than 5 bytes", path.display()))
}

/// WOFF2's known-table list, in the order the spec assigns indices 0..62.
const KNOWN_TABLE_TAGS: [[u8; 4]; 63] = [
    *b"cmap", *b"head", *b"hhea", *b"hmtx", *b"maxp", *b"name", *b"OS/2", *b"post", *b"cvt ",
    *b"fpgm", *b"glyf", *b"loca", *b"prep", *b"CFF ", *b"VORG", *b"EBDT", *b"EBLC", *b"gasp",
    *b"hdmx", *b"kern", *b"LTSH", *b"PCLT", *b"VDMX", *b"vhea", *b"vmtx", *b"BASE", *b"GDEF",
    *b"GPOS", *b"GSUB", *b"EBSC", *b"JSTF", *b"MATH", *b"CBDT", *b"CBLC", *b"COLR", *b"CPAL",
    *b"SVG ", *b"sbix", *b"acnt", *b"avar", *b"bdat", *b"bloc", *b"bsln", *b"cvar", *b"fdsc",
    *b"feat", *b"fmtx", *b"fvar", *b"gvar", *b"hsty", *b"just", *b"lcar", *b"mort", *b"morx",
    *b"opbd", *b"prop", *b"trak", *b"Zapf", *b"Silf", *b"Glat", *b"Gloc", *b"Feat", *b"Sill",
];

/// A `cmap` subtable resolved to a character -> glyph-id function.
enum CmapLookup<'a> {
    /// Format 0: a 256-entry byte array.
    Byte(&'a [u8]),
    /// Format 4: segmented coverage for the BMP.
    Segment(&'a [u8]),
    /// Format 6: a trimmed contiguous array.
    Trimmed(&'a [u8]),
    /// Format 12: segmented coverage over the full range.
    Group(&'a [u8]),
}

impl<'a> CmapLookup<'a> {
    /// fontkit's `CmapProcessor` subtable preference, in order. Anything after the
    /// unicode entries is a legacy encoding, which for a Latin sample string behaves
    /// the same as (3,1).
    const PREFERENCE: [(u16, u16); 10] = [
        (3, 10),
        (0, 6),
        (0, 4),
        (3, 1),
        (0, 3),
        (0, 2),
        (0, 1),
        (0, 0),
        (3, 0),
        (1, 0),
    ];

    fn parse(cmap: &'a [u8], path: &Path) -> Result<Self, String> {
        let num_tables = be_u16(cmap, 2, b"cmap", path)?;
        let mut found: Vec<(u16, u16, usize)> = Vec::new();
        for index in 0..usize::from(num_tables) {
            let base = 4 + index * 8;
            let platform = be_u16(cmap, base, b"cmap", path)?;
            let encoding = be_u16(cmap, base + 2, b"cmap", path)?;
            let offset = be_u32(cmap, base + 4, b"cmap", path)? as usize;
            found.push((platform, encoding, offset));
        }
        let chosen = Self::PREFERENCE
            .iter()
            .find_map(|(platform, encoding)| {
                found.iter().find(|(p, e, _)| p == platform && e == encoding)
            })
            .or_else(|| found.first())
            .ok_or_else(|| format!("{}: the cmap table has no subtables", path.display()))?;
        let subtable = cmap.get(chosen.2..).ok_or_else(|| {
            format!("{}: cmap subtable offset {} is past the table", path.display(), chosen.2)
        })?;
        let format = be_u16(subtable, 0, b"cmap", path)?;
        match format {
            0 => Ok(Self::Byte(subtable)),
            4 => Ok(Self::Segment(subtable)),
            6 => Ok(Self::Trimmed(subtable)),
            12 => Ok(Self::Group(subtable)),
            other => Err(format!(
                "{}: cmap subtable format {other} is not supported. next/font/local needs \
                 the character -> glyph map to average the font's advance widths, which is \
                 what size-adjust is computed from; guessing it would ship a fallback face \
                 with the wrong metrics.",
                path.display()
            )),
        }
    }

    /// The glyph id for a code point, or 0 (`.notdef`, i.e. "no glyph") when unmapped.
    fn glyph(&self, code: u32) -> u16 {
        match self {
            Self::Byte(data) => {
                if code > 255 {
                    return 0;
                }
                data.get(6 + code as usize).map(|b| u16::from(*b)).unwrap_or(0)
            }
            Self::Segment(data) => segment_lookup(data, code),
            Self::Trimmed(data) => {
                let Ok(first) = be_u16(data, 6, b"cmap", Path::new("")) else { return 0 };
                let Ok(count) = be_u16(data, 8, b"cmap", Path::new("")) else { return 0 };
                if code < u32::from(first) || code >= u32::from(first) + u32::from(count) {
                    return 0;
                }
                let index = (code - u32::from(first)) as usize;
                be_u16(data, 10 + index * 2, b"cmap", Path::new("")).unwrap_or(0)
            }
            Self::Group(data) => {
                let Ok(groups) = be_u32(data, 12, b"cmap", Path::new("")) else { return 0 };
                for index in 0..groups as usize {
                    let base = 16 + index * 12;
                    let (Ok(start), Ok(end), Ok(glyph)) = (
                        be_u32(data, base, b"cmap", Path::new("")),
                        be_u32(data, base + 4, b"cmap", Path::new("")),
                        be_u32(data, base + 8, b"cmap", Path::new("")),
                    ) else {
                        return 0;
                    };
                    if code >= start && code <= end {
                        return (glyph + (code - start)) as u16;
                    }
                }
                0
            }
        }
    }
}

/// cmap format 4: `endCode`/`startCode`/`idDelta`/`idRangeOffset` parallel arrays.
fn segment_lookup(data: &[u8], code: u32) -> u16 {
    if code > 0xffff {
        return 0;
    }
    let code = code as u16;
    let quiet = Path::new("");
    let Ok(seg_count_x2) = be_u16(data, 6, b"cmap", quiet) else { return 0 };
    let seg_count = usize::from(seg_count_x2 / 2);
    let end_codes = 14;
    let start_codes = end_codes + seg_count * 2 + 2;
    let id_deltas = start_codes + seg_count * 2;
    let id_range_offsets = id_deltas + seg_count * 2;
    for segment in 0..seg_count {
        let Ok(end) = be_u16(data, end_codes + segment * 2, b"cmap", quiet) else { return 0 };
        if code > end {
            continue;
        }
        let Ok(start) = be_u16(data, start_codes + segment * 2, b"cmap", quiet) else { return 0 };
        if code < start {
            return 0;
        }
        let Ok(delta) = be_u16(data, id_deltas + segment * 2, b"cmap", quiet) else { return 0 };
        let Ok(range_offset) = be_u16(data, id_range_offsets + segment * 2, b"cmap", quiet) else {
            return 0;
        };
        if range_offset == 0 {
            return code.wrapping_add(delta);
        }
        // The glyph id array is addressed RELATIVE to the idRangeOffset slot itself.
        let index = id_range_offsets
            + segment * 2
            + usize::from(range_offset)
            + usize::from(code - start) * 2;
        let Ok(glyph) = be_u16(data, index, b"cmap", quiet) else { return 0 };
        if glyph == 0 {
            return 0;
        }
        return glyph.wrapping_add(delta);
    }
    0
}

fn truncated(path: &Path, what: &str) -> String {
    format!("{}: the file ends inside the {what}", path.display())
}

fn tag_name(tag: &[u8; 4]) -> String {
    String::from_utf8_lossy(tag).into_owned()
}

fn tag_at(bytes: &[u8], offset: usize, path: &Path) -> Result<[u8; 4], String> {
    bytes
        .get(offset..offset + 4)
        .and_then(|slice| <[u8; 4]>::try_from(slice).ok())
        .ok_or_else(|| truncated(path, "table directory"))
}

fn be_u16(bytes: &[u8], offset: usize, table: &[u8; 4], path: &Path) -> Result<u16, String> {
    let slice = bytes.get(offset..offset + 2).ok_or_else(|| short(path, table, offset))?;
    Ok(u16::from_be_bytes([slice[0], slice[1]]))
}

fn be_i16(bytes: &[u8], offset: usize, table: &[u8; 4], path: &Path) -> Result<i16, String> {
    Ok(be_u16(bytes, offset, table, path)? as i16)
}

fn be_u32(bytes: &[u8], offset: usize, table: &[u8; 4], path: &Path) -> Result<u32, String> {
    let slice = bytes.get(offset..offset + 4).ok_or_else(|| short(path, table, offset))?;
    Ok(u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn short(path: &Path, table: &[u8; 4], offset: usize) -> String {
    format!(
        "{}: the `{}` table is too short to read offset {offset}",
        path.display(),
        String::from_utf8_lossy(table),
    )
}

/// A minimal but REAL sfnt: head + hhea + maxp + hmtx + a format-4 cmap covering the
/// ASCII range, built in code so this reader — and `next_font`'s local-font emit above
/// it — are exercised end to end without shipping a binary fixture. `ascent` is 1000,
/// `descent` -200 and `lineGap` 0.
#[cfg(test)]
pub(crate) fn synthetic_font(units_per_em: u16, advance: u16) -> Vec<u8> {
    synthetic_font_covering(units_per_em, advance, 0x0020)
}

/// `start_code` is the first character the cmap covers, so a test can build a font
/// that does NOT have the sample string's letters.
#[cfg(test)]
pub(crate) fn synthetic_font_covering(
    units_per_em: u16,
    advance: u16,
    start_code: u16,
) -> Vec<u8> {
    {
        fn table(tag: &[u8; 4], data: Vec<u8>) -> ([u8; 4], Vec<u8>) {
            (*tag, data)
        }
        let mut head = vec![0u8; 54];
        head[18..20].copy_from_slice(&units_per_em.to_be_bytes());
        let mut hhea = vec![0u8; 36];
        hhea[4..6].copy_from_slice(&1000i16.to_be_bytes());
        hhea[6..8].copy_from_slice(&(-200i16).to_be_bytes());
        hhea[8..10].copy_from_slice(&0i16.to_be_bytes());
        hhea[34..36].copy_from_slice(&2u16.to_be_bytes()); // numberOfHMetrics
        let mut maxp = vec![0u8; 6];
        maxp[4..6].copy_from_slice(&200u16.to_be_bytes()); // numGlyphs
        // hmtx: two long metrics; every glyph past the first shares the second advance.
        let mut hmtx = Vec::new();
        hmtx.extend_from_slice(&0u16.to_be_bytes());
        hmtx.extend_from_slice(&0i16.to_be_bytes());
        hmtx.extend_from_slice(&advance.to_be_bytes());
        hmtx.extend_from_slice(&0i16.to_be_bytes());
        // cmap: one (3,1) format-4 subtable mapping U+0020..U+007E to glyph 1.
        let mut sub: Vec<u8> = Vec::new();
        let seg_count = 2u16;
        sub.extend_from_slice(&4u16.to_be_bytes()); // format
        sub.extend_from_slice(&0u16.to_be_bytes()); // length (unused here)
        sub.extend_from_slice(&0u16.to_be_bytes()); // language
        sub.extend_from_slice(&(seg_count * 2).to_be_bytes());
        sub.extend_from_slice(&0u16.to_be_bytes()); // searchRange
        sub.extend_from_slice(&0u16.to_be_bytes()); // entrySelector
        sub.extend_from_slice(&0u16.to_be_bytes()); // rangeShift
        sub.extend_from_slice(&0x007eu16.to_be_bytes()); // endCode[0]
        sub.extend_from_slice(&0xffffu16.to_be_bytes()); // endCode[1]
        sub.extend_from_slice(&0u16.to_be_bytes()); // reservedPad
        sub.extend_from_slice(&start_code.to_be_bytes()); // startCode[0]
        sub.extend_from_slice(&0xffffu16.to_be_bytes()); // startCode[1]
        // idDelta[0]: glyph = code + delta. A format-4 segment cannot map a whole range
        // onto ONE glyph, so it maps onto consecutive glyphs starting at 1.
        sub.extend_from_slice(&(1u16.wrapping_sub(start_code)).to_be_bytes());
        sub.extend_from_slice(&1u16.to_be_bytes());
        sub.extend_from_slice(&0u16.to_be_bytes()); // idRangeOffset[0]
        sub.extend_from_slice(&0u16.to_be_bytes()); // idRangeOffset[1]
        let mut cmap = Vec::new();
        cmap.extend_from_slice(&0u16.to_be_bytes()); // version
        cmap.extend_from_slice(&1u16.to_be_bytes()); // numTables
        cmap.extend_from_slice(&3u16.to_be_bytes()); // platform
        cmap.extend_from_slice(&1u16.to_be_bytes()); // encoding
        cmap.extend_from_slice(&12u32.to_be_bytes()); // offset
        cmap.extend_from_slice(&sub);

        let tables = vec![
            table(b"cmap", cmap),
            table(b"head", head),
            table(b"hhea", hhea),
            table(b"hmtx", hmtx),
            table(b"maxp", maxp),
        ];
        let mut out = Vec::new();
        out.extend_from_slice(&0x00010000u32.to_be_bytes());
        out.extend_from_slice(&(tables.len() as u16).to_be_bytes());
        out.extend_from_slice(&[0u8; 6]);
        let mut offset = 12 + tables.len() * 16;
        let mut directory = Vec::new();
        let mut body = Vec::new();
        for (tag, data) in &tables {
            directory.extend_from_slice(tag);
            directory.extend_from_slice(&0u32.to_be_bytes());
            directory.extend_from_slice(&(offset as u32).to_be_bytes());
            directory.extend_from_slice(&(data.len() as u32).to_be_bytes());
            offset += data.len();
            body.extend_from_slice(data);
        }
        out.extend_from_slice(&directory);
        out.extend_from_slice(&body);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_hhea_and_head_and_averages_the_sample_string() {
        let font = synthetic_font(1000, 500);
        let metrics = read_metrics_from_bytes(&font, Path::new("synthetic.ttf")).unwrap();
        assert_eq!(metrics.ascent, 1000.0);
        assert_eq!(metrics.descent, -200.0);
        assert_eq!(metrics.line_gap, 0.0);
        assert_eq!(metrics.units_per_em, 1000.0);
        // Every sample character maps to a glyph past the first, so every advance is
        // the second hmtx entry.
        assert_eq!(metrics.az_avg_width, Some(500.0));
    }

    /// Next's sample string is load bearing: the average is taken over exactly these
    /// 43 characters, so a typo silently changes every `size-adjust` diffpack emits.
    #[test]
    fn the_sample_string_matches_nexts() {
        assert_eq!(AVG_CHARACTERS, "aaabcdeeeefghiijklmnnoopqrrssttuvwxyz      ");
        assert_eq!(AVG_CHARACTERS.chars().count(), 43);
    }

    /// A font with no glyph for one of the sample characters gets NO average, which is
    /// how Next ends up with `size-adjust: 100%` rather than a wrong number.
    #[test]
    fn a_font_missing_a_sample_glyph_has_no_average() {
        // A cmap that starts at U+0100 covers none of the sample string's letters.
        let font = synthetic_font_covering(1000, 500, 0x0100);
        let metrics = read_metrics_from_bytes(&font, Path::new("synthetic.ttf")).unwrap();
        assert_eq!(metrics.az_avg_width, None);
    }

    /// A Brotli stream carrying `data` verbatim: one non-last UNCOMPRESSED meta-block
    /// followed by the empty last meta-block. Brotli's own format, so a real decoder
    /// reads it — which lets the WOFF2 path be tested without shipping a font binary or
    /// linking a compressor.
    fn stored_brotli(data: &[u8]) -> Vec<u8> {
        assert!(!data.is_empty() && data.len() <= 0x10000, "one meta-block's worth");
        let mlen = (data.len() - 1) as u32;
        // WBITS=0 (1 bit) | ISLAST=0 (1) | MNIBBLES=0 => 4 nibbles (2) | MLEN-1 (16)
        // | ISUNCOMPRESSED=1 (1), packed LSB-first, then padded to a byte boundary.
        let mut out = vec![
            ((mlen & 0x0f) << 4) as u8,
            ((mlen >> 4) & 0xff) as u8,
            (((mlen >> 12) & 0x0f) | 0x10) as u8,
        ];
        out.extend_from_slice(data);
        // ISLAST=1, ISLASTEMPTY=1.
        out.push(0x03);
        out
    }

    /// Wrap the tables of an sfnt into a WOFF2 container (null transforms throughout),
    /// which is the format `next/font/local` sees most often — cal.com's own
    /// `CalSans-SemiBold.woff2` included.
    fn woff2_of(sfnt: &[u8]) -> Vec<u8> {
        let num_tables = u16::from_be_bytes([sfnt[4], sfnt[5]]);
        let mut directory = Vec::new();
        let mut body = Vec::new();
        for index in 0..usize::from(num_tables) {
            let base = 12 + index * 16;
            let tag: [u8; 4] = sfnt[base..base + 4].try_into().unwrap();
            let offset = u32::from_be_bytes(sfnt[base + 8..base + 12].try_into().unwrap()) as usize;
            let length =
                u32::from_be_bytes(sfnt[base + 12..base + 16].try_into().unwrap()) as usize;
            let known = KNOWN_TABLE_TAGS.iter().position(|k| *k == tag).unwrap() as u8;
            directory.push(known);
            // UIntBase128 of `length` (every table here is under 128 bytes... not
            // necessarily, so encode properly).
            let mut nibbles = Vec::new();
            let mut value = length as u32;
            loop {
                nibbles.push((value & 0x7f) as u8);
                value >>= 7;
                if value == 0 {
                    break;
                }
            }
            for (position, byte) in nibbles.iter().rev().enumerate() {
                let last = position == nibbles.len() - 1;
                directory.push(if last { *byte } else { *byte | 0x80 });
            }
            body.extend_from_slice(&sfnt[offset..offset + length]);
        }
        let compressed = stored_brotli(&body);
        let mut out = Vec::new();
        out.extend_from_slice(b"wOF2");
        out.extend_from_slice(&0x00010000u32.to_be_bytes()); // flavor
        out.extend_from_slice(&0u32.to_be_bytes()); // length
        out.extend_from_slice(&num_tables.to_be_bytes());
        out.extend_from_slice(&0u16.to_be_bytes()); // reserved
        out.extend_from_slice(&(sfnt.len() as u32).to_be_bytes()); // totalSfntSize
        out.extend_from_slice(&(compressed.len() as u32).to_be_bytes());
        out.extend_from_slice(&[0u8; 4]); // major/minor version
        out.extend_from_slice(&[0u8; 20]); // meta/priv offsets and lengths
        assert_eq!(out.len(), 48);
        out.extend_from_slice(&directory);
        out.extend_from_slice(&compressed);
        out
    }

    /// The container `next/font/local` meets in practice: the tables live inside one
    /// Brotli stream, and the reader must reach exactly the same numbers as from the
    /// bare sfnt.
    #[test]
    fn reads_the_same_metrics_through_a_woff2_wrapper() {
        let sfnt = synthetic_font(1000, 500);
        let woff2 = woff2_of(&sfnt);
        assert_eq!(&woff2[0..4], b"wOF2");
        let from_sfnt = read_metrics_from_bytes(&sfnt, Path::new("synthetic.ttf")).unwrap();
        let from_woff2 = read_metrics_from_bytes(&woff2, Path::new("synthetic.woff2")).unwrap();
        assert_eq!(from_sfnt, from_woff2);
        assert_eq!(from_woff2.az_avg_width, Some(500.0));
    }

    #[test]
    fn a_truncated_file_is_a_hard_error_naming_it() {
        let error = read_metrics_from_bytes(&[0, 1, 0, 0, 0, 3], Path::new("broken.ttf"))
            .unwrap_err();
        assert!(error.contains("broken.ttf"), "{error}");
    }
}

