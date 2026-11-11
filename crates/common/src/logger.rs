//! Port of `codon::Logger`, `getLogger`, `pushLogger`, `popLogger`, `codon::Timer`.

use bitflags::bitflags;
use parking_lot::{Mutex, MutexGuard};
use std::io::Write;
use std::time::Instant;

bitflags! {
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct LoggerFlags: u32 {
        const TIME      = 1 << 0;
        const REALIZE   = 1 << 1;
        const TYPECHECK = 1 << 2;
        const IR        = 1 << 3;
        const USER      = 1 << 4;
    }
}

#[derive(Clone, Debug, Default)]
pub struct Logger {
    pub flags: LoggerFlags,
    pub level: i32,
}

impl Logger {
    /// Parse `-log` flag letters: t=time, r=realize, T=typecheck, i=IR, l=user.
    pub fn parse(&mut self, s: &str) {
        if s.contains('t') {
            self.flags |= LoggerFlags::TIME;
        }
        if s.contains('r') {
            self.flags |= LoggerFlags::REALIZE;
        }
        if s.contains('T') {
            self.flags |= LoggerFlags::TYPECHECK;
        }
        if s.contains('i') {
            self.flags |= LoggerFlags::IR;
        }
        if s.contains('l') {
            self.flags |= LoggerFlags::USER;
        }
    }
}

static LOGGERS: Mutex<Vec<Logger>> = Mutex::new(Vec::new());

/// Current logger (C++ `getLogger()`).
pub fn logger() -> parking_lot::MappedMutexGuard<'static, Logger> {
    let mut g = LOGGERS.lock();
    if g.is_empty() {
        g.push(Logger::default());
    }
    MutexGuard::map(g, |v| v.last_mut().unwrap())
}

pub fn push_logger() {
    LOGGERS.lock().push(Logger::default());
}

pub fn pop_logger() -> bool {
    let mut g = LOGGERS.lock();
    if g.is_empty() {
        return false;
    }
    g.pop();
    true
}

/// Write a debug line to the log stream (C++ `DBG`).
pub fn log_line(msg: &str) {
    let _ = writeln!(std::io::stderr(), "{msg}");
}

/// Scoped timer (C++ `codon::Timer` + `TIME(name)`); logs on drop when TIME flag set.
pub struct Timer {
    name: String,
    start: Instant,
    pub logged: bool,
}

impl Timer {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), start: Instant::now(), logged: false }
    }
    pub fn elapsed(&self) -> f64 {
        self.start.elapsed().as_millis() as f64 / 1000.0
    }
    pub fn log(&mut self) {
        if !self.logged {
            crate::log_flag!(TIME, "[T] {} = {:.3}", self.name, self.elapsed());
            self.logged = true;
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        self.log();
    }
}
