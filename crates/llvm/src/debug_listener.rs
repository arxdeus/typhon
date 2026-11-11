//! Port of `codon/compiler/debug_listener.{h,cpp}`.
//!
//! Tracks JIT-compiled objects and their section load addresses so that runtime
//! backtrace PCs can be symbolized (function, file, line, column).
//!
//! PORT-NOTE: C++ has two trackers: `DebugListener` (an MCJIT
//! `JITEventListener`) and `DebugPlugin` (a JITLink `ObjectLinkingLayer::Plugin`).
//! Neither interface exists in the C API. The Rust engine captures each object
//! file through the ORC object transform layer and learns section load
//! addresses from the memory-manager callbacks, then symbolizes with
//! `addr2line`/`gimli` instead of `llvm::symbolize::LLVMSymbolizer`.

use std::borrow::Cow;

use object::{Object, ObjectSection};
use parking_lot::Mutex;

/// A section as loaded in memory.
#[derive(Clone, Debug)]
pub struct LoadedSection {
    pub name: String,
    pub addr: u64,
    pub size: u64,
}

/// `llvm::DILineInfo` subset.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LineInfo {
    pub function: String,
    pub file: String,
    pub line: u32,
    pub column: u32,
}

struct ObjectInfo {
    key: u64,
    owner: usize,
    data: Vec<u8>,
    sections: Vec<LoadedSection>,
}

/// Port of `DebugListener` / `DebugPlugin`.
#[derive(Default)]
pub struct DebugListener {
    objects: Mutex<Vec<ObjectInfo>>,
}

/// Normalize a section name: Mach-O names reported by RTDyld may be
/// `__TEXT,__text`; keep the part after the comma (as `DebugPlugin` does).
fn normalize(name: &str) -> &str {
    match name.find(',') {
        Some(i) => &name[i + 1..],
        None => name,
    }
}

fn is_text(name: &str) -> bool {
    matches!(normalize(name), ".text" | "__text")
}

#[derive(Debug, Default, Clone)]
struct Reloc(std::sync::Arc<object::read::RelocationMap>);

impl gimli::read::Relocate for Reloc {
    fn relocate_address(&self, offset: usize, value: u64) -> gimli::Result<u64> {
        Ok(self.0.relocate(offset as u64, value))
    }
    fn relocate_offset(&self, offset: usize, value: usize) -> gimli::Result<usize> {
        <usize as gimli::ReaderOffset>::from_u64(self.0.relocate(offset as u64, value as u64))
    }
}

type Reader<'a> = gimli::RelocateReader<gimli::EndianSlice<'a, gimli::RunTimeEndian>, Reloc>;

fn symbolize_object(data: &[u8], probe: u64) -> Option<LineInfo> {
    let file = object::File::parse(data).ok()?;
    let endian = if file.is_little_endian() { gimli::RunTimeEndian::Little } else { gimli::RunTimeEndian::Big };
    let mut owned: Vec<Cow<'_, [u8]>> = Vec::new();
    let mut maps: Vec<Reloc> = Vec::new();
    let ids: Vec<gimli::SectionId> = vec![
        gimli::SectionId::DebugAbbrev,
        gimli::SectionId::DebugAddr,
        gimli::SectionId::DebugAranges,
        gimli::SectionId::DebugInfo,
        gimli::SectionId::DebugLine,
        gimli::SectionId::DebugLineStr,
        gimli::SectionId::DebugStr,
        gimli::SectionId::DebugStrOffsets,
        gimli::SectionId::DebugTypes,
        gimli::SectionId::DebugLoc,
        gimli::SectionId::DebugLocLists,
        gimli::SectionId::DebugRanges,
        gimli::SectionId::DebugRngLists,
    ];
    for id in &ids {
        let section = file.section_by_name(id.name()).or_else(|| id.xcoff_name().and_then(|n| file.section_by_name(n)));
        match section {
            Some(s) => {
                owned.push(s.uncompressed_data().unwrap_or(Cow::Borrowed(&[])));
                maps.push(Reloc(std::sync::Arc::new(s.relocation_map().unwrap_or_default())));
            }
            None => {
                owned.push(Cow::Borrowed(&[]));
                maps.push(Reloc::default());
            }
        }
    }
    let load = |id: gimli::SectionId| -> Result<Reader<'_>, ()> {
        let i = ids.iter().position(|x| *x == id);
        Ok(match i {
            Some(i) => gimli::RelocateReader::new(gimli::EndianSlice::new(&owned[i], endian), maps[i].clone()),
            None => gimli::RelocateReader::new(gimli::EndianSlice::new(&[], endian), Reloc::default()),
        })
    };
    let dwarf = gimli::Dwarf::load(load).ok()?;
    let ctx = addr2line::Context::from_dwarf(dwarf).ok()?;
    let mut info = LineInfo::default();
    let mut frames = ctx.find_frames(probe).skip_all_loads().ok()?;
    if let Ok(Some(frame)) = frames.next() {
        if let Some(f) = frame.function.as_ref() {
            info.function = f.demangle().map(|c| c.into_owned()).unwrap_or_default();
        }
        if let Some(loc) = frame.location {
            info.file = loc.file.unwrap_or("").to_string();
            info.line = loc.line.unwrap_or(0);
            info.column = loc.column.unwrap_or(0);
        }
    }
    if info.function.is_empty() {
        // Fall back to the symbol table (no debug info).
        let mut best: Option<(u64, String)> = None;
        for sym in file.symbols() {
            use object::ObjectSymbol;
            if sym.kind() != object::SymbolKind::Text {
                continue;
            }
            let a = sym.address();
            if a <= probe && best.as_ref().is_none_or(|(b, _)| a > *b) {
                best = Some((a, sym.name().unwrap_or("").to_string()));
            }
        }
        if let Some((_, n)) = best {
            info.function = n.trim_start_matches('_').to_string();
            if info.function.is_empty() {
                return None;
            }
        } else {
            return None;
        }
    }
    Some(info)
}

/// Port of `runtime::makeBacktraceFrameString`.
pub fn make_backtrace_frame_string(pc: usize, func: &str, file: &str, line: u32, col: u32) -> String {
    let mut s = format!("[\x1b[33m0x{pc:x}\x1b[0m]");
    if !func.is_empty() {
        s += &format!(" \x1b[32m{func}\x1b[0m");
        if !file.is_empty() {
            s += &format!(" at \x1b[36m{file}\x1b[0m");
            if line != 0 {
                s += &format!(":\x1b[33m{line}\x1b[0m");
                if col != 0 {
                    s += &format!(":\x1b[33m{col}\x1b[0m");
                }
            }
        }
    }
    s
}

impl DebugListener {
    pub fn new() -> Self {
        Self::default()
    }

    /// `notifyObjectLoaded` / `DebugPlugin::notifyEmitted`.
    pub(crate) fn register(&self, key: u64, data: Vec<u8>, sections: Vec<LoadedSection>, owner: usize) {
        self.objects.lock().push(ObjectInfo { key, owner, data, sections });
    }

    /// `notifyFreeingObject` / `notifyRemovingResources`.
    pub(crate) fn unregister_owner(&self, owner: usize) {
        self.objects.lock().retain(|o| o.owner != owner);
    }

    pub fn unregister(&self, key: u64) {
        self.objects.lock().retain(|o| o.key != key);
    }

    /// Number of tracked objects.
    pub fn len(&self) -> usize {
        self.objects.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Port of `symbolize(pc)`. Returns `None` if `pc` is not in JIT code.
    pub fn symbolize(&self, pc: usize) -> Option<LineInfo> {
        let objects = self.objects.lock();
        let pc = pc as u64;
        for o in objects.iter() {
            for s in &o.sections {
                if !is_text(&s.name) || !(s.addr <= pc && pc < s.addr + s.size) {
                    continue;
                }
                // Translate to an object-relative address: offset in the text
                // section + the section's address in the object file.
                let file = object::File::parse(&*o.data).ok()?;
                let sec_addr = file
                    .sections()
                    .find(|sec| sec.name().map(|n| normalize(n) == normalize(&s.name)).unwrap_or(false))
                    .map(|sec| sec.address())
                    .unwrap_or(0);
                return symbolize_object(&o.data, pc - s.addr + sec_addr);
            }
        }
        None
    }

    /// Port of `getPrettyBacktrace(pc)`: empty if unknown.
    pub fn pretty_backtrace_frame(&self, pc: usize) -> String {
        match self.symbolize(pc) {
            Some(src) if src.function != "<invalid>" && src.file != "<invalid>" => {
                make_backtrace_frame_string(pc, &src.function, &src.file, src.line, src.column)
            }
            _ => String::new(),
        }
    }

    /// Port of `getPrettyBacktrace(backtrace)`.
    pub fn pretty_backtrace(&self, backtrace: &[usize]) -> String {
        let mut buf = String::from("\x1b[1mBacktrace:\x1b[0m\n");
        for &pc in backtrace {
            let line = self.pretty_backtrace_frame(pc);
            if !line.is_empty() {
                buf += "  ";
                buf += &line;
                buf += "\n";
            }
        }
        buf
    }
}
