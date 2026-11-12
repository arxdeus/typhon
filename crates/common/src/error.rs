//! .
//!
//! the reference implementation uses exceptions (`exc::ParserException`) plus `llvm::Error` wrappers
//! (`ParserErrorInfo`, `RuntimeErrorInfo`, `PluginErrorInfo`, `IOErrorInfo`).
//! Rust uses a single `thiserror` enum [`CompilerError`] carried in `Result`.

use crate::srcinfo::SrcInfo;
use std::fmt::{self, Display};

pub type Result<T, E = CompilerError> = std::result::Result<T, E>;

/// Discriminants match the the reference implementation enum order (used as
/// error codes in `https://upstream.io/error/{:04}` links).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
#[allow(non_camel_case_types)]
pub enum ErrorKind {
    CALL_NAME_ORDER, CALL_NAME_STAR, CALL_ELLIPSIS, IMPORT_IDENTIFIER, IMPORT_FN,
    IMPORT_STAR, FN_LLVM, FN_LAST_KWARG, FN_MULTIPLE_ARGS, FN_DEFAULT_STARARG,
    FN_ARG_TWICE, FN_DEFAULT, FN_C_DEFAULT, FN_C_TYPE, FN_SINGLE_DECORATOR,
    CLASS_EXTENSION, CLASS_MISSING_TYPE, CLASS_ARG_TWICE, CLASS_BAD_DECORATOR,
    CLASS_MULTIPLE_DECORATORS, CLASS_SINGLE_DECORATOR, CLASS_CONFLICT_DECORATOR,
    CLASS_NONSTATIC_DECORATOR, CLASS_BAD_DECORATOR_ARG, ID_NOT_FOUND, ID_CANNOT_CAPTURE,
    ID_INVALID_BIND, UNION_TOO_BIG, COMPILER_NO_FILE, COMPILER_NO_STDLIB, ID_NONLOCAL,
    IMPORT_NO_MODULE, IMPORT_NO_NAME, DEL_NOT_ALLOWED, DEL_INVALID, ASSIGN_INVALID,
    ASSIGN_LOCAL_REFERENCE, ASSIGN_MULTI_STAR, INT_RANGE, FLOAT_RANGE,
    STR_FSTRING_BALANCE_EXTRA, STR_FSTRING_BALANCE_MISSING, CALL_NO_TYPE,
    CALL_TUPLE_COMPREHENSION, CALL_NAMEDTUPLE, CALL_PARTIAL, EXPECTED_TOPLEVEL,
    CLASS_ID_NOT_FOUND, CLASS_INVALID_BIND, CLASS_NO_INHERIT, CLASS_NO_EXTEND,
    CLASS_TUPLE_INHERIT, CLASS_BAD_MRO, CLASS_BAD_ATTR, MATCH_MULTI_ELLIPSIS,
    FN_OUTSIDE_ERROR, FN_GLOBAL_ASSIGNED, FN_GLOBAL_NOT_FOUND, FN_BAD_LLVM,
    FN_REALIZE_BUILTIN, EXPECTED_LOOP, LOOP_DECORATOR, BAD_STATIC_TYPE, EXPECTED_TYPE,
    UNEXPECTED_TYPE, DOT_NO_ATTR, DOT_NO_ATTR_ARGS, FN_NO_ATTR_ARGS, EXPECTED_STATIC,
    EXPECTED_STATIC_SPECIFIED, ASSIGN_UNEXPECTED_STATIC, ASSIGN_UNEXPECTED_FROZEN,
    CALL_BAD_UNPACK, CALL_BAD_ITER, CALL_BAD_KWUNPACK, CALL_REPEATED_NAME,
    CALL_RECURSIVE_DEFAULT, CALL_SUPERF, CALL_SUPER_PARENT, CALL_PTR_VAR, EXPECTED_TUPLE,
    CALL_REALIZED_FN, CALL_ARGS_MANY, CALL_ARGS_INVALID, CALL_ARGS_MISSING,
    GENERICS_MISMATCH, EXPECTED_GENERATOR, STATIC_RANGE_BOUNDS, TUPLE_RANGE_BOUNDS,
    STATIC_DIV_ZERO, SLICE_STEP_ZERO, OP_NO_MAGIC, INST_CALLABLE_STATIC,
    CATCH_EXCEPTION_TYPE, TYPE_CANNOT_REALIZE_ATTR, TYPE_UNIFY, TYPE_FAILED,
    MAX_REALIZATION, CUSTOM,
}

/// Substitute positional `{}` placeholders (`fmt::format(fmt::runtime(..), args...)`).
/// `{{` / `}}` are literal braces.
pub fn format_runtime(fmt_str: &str, args: &[&dyn Display]) -> String {
    let mut out = String::with_capacity(fmt_str.len() + 16);
    let mut it = args.iter();
    let mut chars = fmt_str.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
                out.push('{');
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
                out.push('}');
            }
            '{' if chars.peek() == Some(&'}') => {
                chars.next();
                if let Some(a) = it.next() {
                    out.push_str(&a.to_string());
                }
            }
            c => out.push(c),
        }
    }
    out
}

/// Port of `Emsg(Error, args...)`: render the human-readable message.
pub fn emsg(e: ErrorKind, args: &[&dyn Display]) -> String {
    use ErrorKind::*;
    let f = |s: &str| format_runtime(s, args);
    match e {
        CALL_NAME_ORDER => f("positional argument follows keyword argument"),
        CALL_NAME_STAR => f("cannot use starred expression here"),
        CALL_ELLIPSIS => f("multiple ellipsis expressions"),
        IMPORT_IDENTIFIER => f("expected identifier"),
        IMPORT_FN => f("function signatures only allowed when importing C or Python functions"),
        IMPORT_STAR => f("import * only allowed at module level"),
        FN_LLVM => f("return types required for LLVM and C functions"),
        FN_LAST_KWARG => f("kwargs must be the last argument"),
        FN_MULTIPLE_ARGS => f("multiple star arguments provided"),
        FN_DEFAULT_STARARG => f("star arguments cannot have default values"),
        FN_ARG_TWICE => f("duplicate argument '{}' in function definition"),
        FN_DEFAULT => f("non-default argument '{}' follows default argument"),
        FN_C_DEFAULT => f("argument '{}' within C function definition cannot have default value"),
        FN_C_TYPE => f("argument '{}' within C function definition requires type annotation"),
        FN_SINGLE_DECORATOR => f("cannot combine '@{}' with other attributes or decorators"),
        CLASS_EXTENSION => f("class extensions cannot define data attributes and generics or inherit other classes"),
        CLASS_MISSING_TYPE => f("type required for data attribute '{}'"),
        CLASS_ARG_TWICE => f("duplicate data attribute '{}' in class definition"),
        CLASS_BAD_DECORATOR => f("unsupported class decorator"),
        CLASS_MULTIPLE_DECORATORS => f("duplicate decorator '@{}' in class definition"),
        CLASS_SINGLE_DECORATOR => f("cannot combine '@{}' with other attributes or decorators"),
        CLASS_CONFLICT_DECORATOR => f("cannot combine '@{}' with '@{}'"),
        CLASS_NONSTATIC_DECORATOR => f("class decorator arguments must be compile-time static values"),
        CLASS_BAD_DECORATOR_ARG => f("class decorator got unexpected argument"),
        ID_NOT_FOUND => f("name '{}' is not defined"),
        ID_CANNOT_CAPTURE => f("name '{}' cannot be captured"),
        ID_NONLOCAL => f("no binding for nonlocal '{}' found"),
        ID_INVALID_BIND => f("cannot bind '{}' to global or nonlocal name"),
        IMPORT_NO_MODULE => f("no module named '{}'"),
        IMPORT_NO_NAME => f("cannot import name '{}' from '{}'"),
        DEL_NOT_ALLOWED => f("name '{}' cannot be deleted"),
        DEL_INVALID => f("cannot delete given expression"),
        ASSIGN_INVALID => f("cannot assign to given expression"),
        ASSIGN_LOCAL_REFERENCE => f("local variable '{}' referenced before assignment at {}"),
        ASSIGN_MULTI_STAR => f("multiple starred expressions in assignment"),
        INT_RANGE => f("integer '{}' cannot fit into 64-bit integer"),
        FLOAT_RANGE => f("float '{}' cannot fit into 64-bit float"),
        STR_FSTRING_BALANCE_EXTRA => f("expecting '}}' in f-string"),
        STR_FSTRING_BALANCE_MISSING => f("single '}}' is not allowed in f-string"),
        CALL_NO_TYPE => f("cannot use calls in type signatures"),
        CALL_TUPLE_COMPREHENSION => f("tuple constructor does not accept nested or conditioned comprehensions"),
        CALL_NAMEDTUPLE => f("namedtuple() takes 2 static arguments"),
        CALL_PARTIAL => f("partial() takes 1 or more arguments"),
        EXPECTED_TOPLEVEL => f("{} must be a top-level statement"),
        CLASS_ID_NOT_FOUND => f("class name '{}' is not defined"),
        CLASS_INVALID_BIND => f("cannot bind '{}' to class or function"),
        CLASS_NO_INHERIT => f("{} classes cannot inherit {} classes"),
        CLASS_NO_EXTEND => f("'{}' cannot be extended"),
        CLASS_TUPLE_INHERIT => f("reference classes cannot inherit tuple classes"),
        CLASS_BAD_MRO => f("inconsistent class hierarchy"),
        CLASS_BAD_ATTR => f("unexpected expression in class definition"),
        MATCH_MULTI_ELLIPSIS => f("multiple ellipses in a pattern"),
        FN_OUTSIDE_ERROR => f("'{}' outside function"),
        FN_GLOBAL_ASSIGNED => f("name '{}' is assigned to before global declaration"),
        FN_GLOBAL_NOT_FOUND => f("no binding for {} '{}' found"),
        FN_BAD_LLVM => f("invalid LLVM code"),
        FN_REALIZE_BUILTIN => f("builtin, exported and external functions cannot be generic"),
        EXPECTED_LOOP => f("'{}' outside loop"),
        LOOP_DECORATOR => f("invalid loop decorator"),
        BAD_STATIC_TYPE => f("expected 'int', 'bool' or 'str'"),
        EXPECTED_TYPE => f("expected {} expression"),
        UNEXPECTED_TYPE => f("unexpected {} expression"),
        UNION_TOO_BIG => f("union exceeded its maximum capacity (contains more than {} types)"),
        DOT_NO_ATTR => f("'{}' object has no attribute '{}'"),
        DOT_NO_ATTR_ARGS => f("'{}' object has no method '{}' with arguments {}"),
        FN_NO_ATTR_ARGS => f("no function '{}' with arguments {}"),
        EXPECTED_STATIC => f("expected static expression"),
        EXPECTED_STATIC_SPECIFIED => f("expected static {} expression"),
        ASSIGN_UNEXPECTED_STATIC => f("cannot modify static expressions"),
        ASSIGN_UNEXPECTED_FROZEN => f("cannot modify tuple attributes"),
        CALL_BAD_UNPACK => f("argument after * must be a tuple, not '{}'"),
        CALL_BAD_ITER => f("iterable must be a tuple, not '{}'"),
        CALL_BAD_KWUNPACK => f("argument after ** must be a named tuple, not '{}'"),
        CALL_REPEATED_NAME => f("keyword argument repeated: {}"),
        CALL_RECURSIVE_DEFAULT => f("argument '{}' has recursive default value"),
        CALL_SUPERF => f("no superf methods found"),
        CALL_SUPER_PARENT => f("no super methods found"),
        CALL_PTR_VAR => f("__ptr__() only takes identifiers or fields rooted in identifiers as arguments"),
        EXPECTED_TUPLE => f("expected tuple type"),
        CALL_REALIZED_FN => f("static.realized() only takes functions as a first argument"),
        CALL_ARGS_MANY => f("{}() takes {} arguments ({} given)"),
        CALL_ARGS_INVALID => f("'{}' is an invalid keyword argument for {}()"),
        CALL_ARGS_MISSING => f("{}() missing 1 required positional argument: '{}'"),
        GENERICS_MISMATCH => f("{} takes {} generics ({} given)"),
        EXPECTED_GENERATOR => f("expected iterable expression"),
        STATIC_RANGE_BOUNDS => f("static.range too large (expected 0..{}, got instead {})"),
        TUPLE_RANGE_BOUNDS => f("tuple index out of range (expected 0..{}, got instead {})"),
        STATIC_DIV_ZERO => f("static division by zero"),
        SLICE_STEP_ZERO => f("slice step cannot be zero"),
        OP_NO_MAGIC => f("unsupported operand type(s) for {}: '{}' and '{}'"),
        INST_CALLABLE_STATIC => f("CallableTrait cannot take static types"),
        CATCH_EXCEPTION_TYPE => f("'{}' does not inherit from BaseException"),
        TYPE_CANNOT_REALIZE_ATTR => f("type of attribute '{}' of object '{}' cannot be inferred"),
        TYPE_UNIFY => f("'{}' does not match expected type '{}'"),
        TYPE_FAILED => f("cannot infer the complete type of an expression (inferred only '{}')"),
        COMPILER_NO_FILE => f("cannot open file '{}' for parsing"),
        COMPILER_NO_STDLIB => f("cannot locate standard library"),
        MAX_REALIZATION => f("maximum realization depth reached during the realization of '{}'"),
        // the reference implementation `Eformat(args...)`: first arg is the format string, rest are its args.
        CUSTOM => match args.split_first() {
            Some((fmt0, rest)) => format_runtime(&fmt0.to_string(), rest),
            None => String::new(),
        },
    }
}

#[derive(Clone, Debug)]
pub struct ErrorMessage {
    pub msg: String,
    pub loc: SrcInfo,
    pub error_code: i32,
}

impl ErrorMessage {
    pub fn new(msg: impl Into<String>, loc: SrcInfo) -> Self {
        Self { msg: msg.into(), loc, error_code: -1 }
    }
    pub fn with_code(msg: impl Into<String>, loc: SrcInfo, code: i32) -> Self {
        Self { msg: msg.into(), loc, error_code: code }
    }
    pub fn file(&self) -> &str {
        self.loc.file.as_str()
    }
}

impl PartialEq for ErrorMessage {
    fn eq(&self, o: &Self) -> bool {
        self.msg == o.msg && self.loc == o.loc
    }
}

impl Display for ErrorMessage {
    /// the reference implementation `ErrorMessage::toString()`: `file:line:col: msg`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.file().is_empty() {
            write!(f, "{}", self.file())?;
            if self.loc.line != 0 {
                write!(f, ":{}", self.loc.line)?;
                if self.loc.col != 0 {
                    write!(f, ":{}", self.loc.col)?;
                }
            }
            write!(f, ": ")?;
        }
        write!(f, "{}", self.msg)
    }
}

/// One error with its note/backtrace chain .
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Backtrace {
    pub trace: Vec<ErrorMessage>,
}

impl Backtrace {
    pub fn add_message(&mut self, msg: impl Into<String>, info: SrcInfo) {
        self.trace.push(ErrorMessage::new(msg, info));
    }
}

#[derive(Clone, Debug, Default)]
pub struct ParserErrors {
    pub errors: Vec<Backtrace>,
}

impl ParserErrors {
    pub fn from_message(m: ErrorMessage) -> Self {
        Self { errors: vec![Backtrace { trace: vec![m] }] }
    }
    pub fn from_messages(ms: Vec<ErrorMessage>) -> Self {
        Self { errors: ms.into_iter().map(|m| Backtrace { trace: vec![m] }).collect() }
    }
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }
    pub fn add_error(&mut self, bt: Backtrace) {
        if self.errors.last() != Some(&bt) {
            self.errors.push(bt);
        }
    }
    pub fn append(&mut self, other: &ParserErrors) {
        for bt in &other.errors {
            self.add_error(bt.clone());
        }
    }
    /// First message of first error (`getMessage()`).
    pub fn message(&self) -> String {
        self.errors
            .first()
            .and_then(|b| b.trace.first())
            .map(|m| m.msg.clone())
            .unwrap_or_default()
    }
}

impl Display for ParserErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for bt in &self.errors {
            for m in &bt.trace {
                writeln!(f, "{m}")?;
            }
        }
        Ok(())
    }
}

/// Unified compiler error (replaces `exc::ParserException` and the `llvm::ErrorInfo`
/// subclasses `ParserErrorInfo`, `RuntimeErrorInfo`, `PluginErrorInfo`, `IOErrorInfo`).
#[derive(Debug, Clone, thiserror::Error)]
pub enum CompilerError {
    #[error("{0}")]
    Parser(ParserErrors),
    #[error("{kind}: {message}")]
    Runtime {
        output: String,
        kind: String,
        message: ErrorMessage,
        backtrace: Vec<String>,
    },
    #[error("{0}")]
    Plugin(String),
    #[error("{0}")]
    Io(String),
    #[error("LLVM error: {0}")]
    Llvm(String),
}

impl CompilerError {
    /// the reference implementation `E(Error, SrcInfo, args...)` without the throw.
    pub fn parser(kind: ErrorKind, src: SrcInfo, args: &[&dyn Display]) -> Self {
        let msg = emsg(kind, args);
        CompilerError::Parser(ParserErrors::from_message(ErrorMessage::with_code(
            msg,
            src,
            kind as i32,
        )))
    }
    pub fn custom(msg: impl Into<String>, src: SrcInfo) -> Self {
        CompilerError::Parser(ParserErrors::from_message(ErrorMessage::new(msg, src)))
    }
    pub fn parser_errors(&self) -> Option<&ParserErrors> {
        match self {
            CompilerError::Parser(p) => Some(p),
            _ => None,
        }
    }
    pub fn parser_errors_mut(&mut self) -> Option<&mut ParserErrors> {
        match self {
            CompilerError::Parser(p) => Some(p),
            _ => None,
        }
    }
}

impl From<std::io::Error> for CompilerError {
    fn from(e: std::io::Error) -> Self {
        CompilerError::Io(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_messages() {
        let s = emsg(ErrorKind::DOT_NO_ATTR, &[&"int", &"foo"]);
        assert_eq!(s, "'int' object has no attribute 'foo'");
        assert_eq!(emsg(ErrorKind::STR_FSTRING_BALANCE_EXTRA, &[]), "expecting '}' in f-string");
        assert_eq!(emsg(ErrorKind::CUSTOM, &[&"x={} y={}", &1, &2]), "x=1 y=2");
    }

    #[test]
    fn error_codes_match_cpp_order() {
        assert_eq!(ErrorKind::CALL_NAME_ORDER as i32, 0);
        assert_eq!(ErrorKind::ID_NOT_FOUND as i32, 24);
        assert_eq!(ErrorKind::CUSTOM as i32, 98);
    }
}
