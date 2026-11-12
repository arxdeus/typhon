//! Shared foundation for the Typhon compiler.

pub mod config;
pub mod diag;
pub mod error;
pub mod fs;
pub mod logger;
pub mod mangle;
pub mod options;
pub mod srcinfo;
pub mod strings;
pub mod target;

pub use error::{CompilerError, ErrorKind, ErrorMessage, ParserErrors, Result, emsg};
pub use logger::{Logger, LoggerFlags, Timer, logger};
pub use options::Options;
pub use srcinfo::{SourceFile, SrcInfo, SrcObject};

/// `seqassert` / `seqassertn`: debug-only internal invariant check.
#[macro_export]
macro_rules! seqassert {
    ($cond:expr, $($arg:tt)+) => {
        debug_assert!($cond, "assertion failed: {}: {}", stringify!($cond), format!($($arg)+));
    };
}

/// `E(error::Error, SrcInfo, args...)`: build a parser error and return it
/// as `Err(CompilerError::Parser(..))` from the current function.
///
/// the reference throws `exc::ParserException`; Rust code returns `Result<T, CompilerError>`.
#[macro_export]
macro_rules! bail_e {
    ($kind:expr, $src:expr $(, $arg:expr)* $(,)?) => {
        return Err($crate::error::CompilerError::parser($kind, $src.clone(), &[$(&$arg as &dyn std::fmt::Display),*]))
    };
}

/// `LOG_TIME`, `LOG_REALIZE`, `LOG_TYPECHECK`, `LOG_IR`, `LOG_USER`.
#[macro_export]
macro_rules! log_flag {
    ($flag:ident, $($arg:tt)+) => {
        if $crate::logger::logger().flags.contains($crate::logger::LoggerFlags::$flag) {
            $crate::logger::log_line(&format!($($arg)+));
        }
    };
}
