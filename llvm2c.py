#!/usr/bin/env python3
"""
llvm2c – Convert LLVM IR (.ll) to C source code.

Usage::

    python llvm2c.py input.ll [-o output.c]
    python llvm2c.py - < input.ll        # read from stdin
"""

import sys
import re
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------

_INT_TYPES: Dict[str, str] = {
    "i1": "bool",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "i128": "__int128",  # GCC/Clang extension; not available on all compilers
}

_FLOAT_TYPES: Dict[str, str] = {
    "half": "float",
    "float": "float",
    "double": "double",
    "x86_fp80": "long double",
    "fp128": "long double",
    "ppc_fp128": "long double",
}


def llvm_type_to_c(t: str) -> str:
    """Return the C type string for an LLVM type string *t*."""
    t = t.strip()
    if t == "void":
        return "void"
    if t in _INT_TYPES:
        return _INT_TYPES[t]
    if t in _FLOAT_TYPES:
        return _FLOAT_TYPES[t]
    if t == "ptr":
        return "void*"
    # Old-style typed pointer  T*
    if t.endswith("*"):
        return llvm_type_to_c(t[:-1].strip()) + "*"
    # Array  [N x T]
    m = re.match(r"^\[(\d+)\s+x\s+(.+)\]$", t)
    if m:
        return f"{llvm_type_to_c(m.group(2).strip())}[{m.group(1)}]"
    # Vector  <N x T>
    m = re.match(r"^<(\d+)\s+x\s+(.+)>$", t)
    if m:
        return f"{llvm_type_to_c(m.group(2).strip())}[{m.group(1)}]"
    # Named struct  %Name
    if t.startswith("%"):
        return re.sub(r"[^a-zA-Z0-9_]", "_", t[1:])
    # Anonymous struct / union
    if t.startswith("{"):
        return "struct { void* _data; }"
    return f"/* {t} */ void*"


def c_decl(c_type: str, name: str) -> str:
    """Build ``c_type name``, moving array brackets after *name* as C requires.

    >>> c_decl("int32_t[10]", "arr")
    'int32_t arr[10]'
    """
    m = re.match(r"^(.+?)(\[[\d\[\]]*\])$", c_type)
    if m:
        return f"{m.group(1)} {name}{m.group(2)}"
    return f"{c_type} {name}"


# ---------------------------------------------------------------------------
# Name / value helpers
# ---------------------------------------------------------------------------

def sanitize(name: str) -> str:
    """Strip the ``%``/``@`` sigil and turn *name* into a valid C identifier."""
    if name and name[0] in ("%", "@"):
        name = name[1:]
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if name and name[0].isdigit():
        name = "_" + name
    return name or "_anon"


def llvm_val(v: str) -> str:
    """Convert a single LLVM value token to a C expression fragment."""
    v = v.strip()
    if v in ("null", "nullptr"):
        return "NULL"
    if v == "true":
        return "1"
    if v == "false":
        return "0"
    if v in ("undef", "poison"):
        return "0 /*undef*/"
    if v == "zeroinitializer":
        return "{0}"
    # Integer / float literal (possibly hex)
    if re.match(r"^-?(?:0x[0-9a-fA-F]+|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)$", v):
        return v
    if v.startswith("%") or v.startswith("@"):
        return sanitize(v)
    return v


# ---------------------------------------------------------------------------
# Comma-splitting that respects nested brackets / angle-brackets
# ---------------------------------------------------------------------------

def split_comma(s: str) -> List[str]:
    """Split *s* on commas while ignoring commas inside ``([{<`` … ``>}])``."""
    parts: List[str] = []
    depth = 0
    cur: List[str] = []
    for ch in s:
        if ch in "([{<":
            depth += 1
            cur.append(ch)
        elif ch in ")]}>":  
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return parts


# ---------------------------------------------------------------------------
# Parameter-attribute stripping
# ---------------------------------------------------------------------------

_PARAM_ATTR_RE = re.compile(
    r"\b(?:noundef|nonnull|noalias|nocapture|readonly|readnone|writeonly|"
    r"zeroext|signext|inreg|byval|inalloca|sret|returned|nofree|nest|"
    r"immarg|swiftself|swifterror|align|dereferenceable|"
    r"dereferenceable_or_null)(?:\s*\(\d+\))?\s*",
    re.IGNORECASE,
)


def strip_param_attrs(s: str) -> str:
    """Remove parameter attributes from a parameter declaration fragment."""
    return _PARAM_ATTR_RE.sub("", s).strip()


# ---------------------------------------------------------------------------
# IR data structures
# ---------------------------------------------------------------------------

@dataclass
class Instr:
    result: Optional[str]   # dest variable name (without ``%``), or None
    opcode: str
    raw_rest: str           # everything after the opcode keyword


@dataclass
class Block:
    label: str
    instrs: List[Instr] = field(default_factory=list)


@dataclass
class Function:
    name: str
    ret_type: str
    params: List[Tuple[str, str]]   # (llvm_type, c_name)
    blocks: List[Block] = field(default_factory=list)
    is_decl: bool = False


@dataclass
class GlobalVar:
    name: str
    ty: str
    init: Optional[str]
    is_const: bool


@dataclass
class Module:
    functions: List[Function] = field(default_factory=list)
    globals: List[GlobalVar] = field(default_factory=list)
    type_defs: Dict[str, str] = field(default_factory=dict)  # name → llvm body


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class Parser:
    """Line-oriented LLVM IR textual-format parser."""

    def __init__(self, src: str) -> None:
        self._lines = src.splitlines()
        self._pos = 0

    # -- internal helpers ---------------------------------------------------

    def _skip_blank(self) -> None:
        while self._pos < len(self._lines):
            s = self._lines[self._pos].strip()
            if s and not s.startswith(";"):
                return
            self._pos += 1

    def _peek(self) -> Optional[str]:
        self._skip_blank()
        return self._lines[self._pos].strip() if self._pos < len(self._lines) else None

    def _next(self) -> Optional[str]:
        self._skip_blank()
        if self._pos < len(self._lines):
            line = self._lines[self._pos].strip()
            self._pos += 1
            return line
        return None

    # -- top-level ----------------------------------------------------------

    def parse(self) -> Module:
        mod = Module()
        while True:
            line = self._next()
            if line is None:
                break
            if line.startswith("define "):
                f = self._parse_func_def(line)
                if f:
                    mod.functions.append(f)
            elif line.startswith("declare "):
                f = self._parse_func_decl(line)
                if f:
                    mod.functions.append(f)
            elif re.match(r"^%[\w.]+\s*=\s*type\s+", line):
                self._parse_type_def(line, mod)
            elif re.match(r"^@[\w.]+\s*=", line):
                gv = self._parse_global(line)
                if gv:
                    mod.globals.append(gv)
            # target triple / datalayout / attributes / metadata → skip
        return mod

    # -- type definitions ---------------------------------------------------

    def _parse_type_def(self, line: str, mod: Module) -> None:
        m = re.match(r"^(%[\w.]+)\s*=\s*type\s+(.+)$", line)
        if m:
            mod.type_defs[m.group(1)[1:]] = m.group(2).strip()

    # -- global variables ---------------------------------------------------

    _GLOBAL_KEYWORDS = re.compile(
        r"\b(?:private|internal|available_externally|linkonce|weak|common|"
        r"appending|extern_weak|linkonce_odr|weak_odr|external|"
        r"dso_local|unnamed_addr|local_unnamed_addr|"
        r"constant|global|hidden|protected|default|"
        r"addrspace\(\d+\))\b\s*"
    )

    def _parse_global(self, line: str) -> Optional[GlobalVar]:
        m = re.match(r"^(@[\w.]+)\s*=\s*(.+)$", line)
        if not m:
            return None
        name = sanitize(m.group(1))
        rest = m.group(2).strip()
        is_const = "constant" in rest
        rest = self._GLOBAL_KEYWORDS.sub("", rest).strip()
        rest = re.sub(r",\s*align\s+\d+\s*$", "", rest).strip()
        parts = rest.split(None, 1)
        ty = parts[0] if parts else "i8"
        init = parts[1].strip() if len(parts) > 1 else None
        return GlobalVar(name=name, ty=ty, init=init, is_const=is_const)

    # -- function declarations ----------------------------------------------

    def _parse_func_decl(self, line: str) -> Optional[Function]:
        m = re.match(r"^declare\s+(.+?)\s+(@[\w.]+)\s*\(([^)]*)\)", line)
        if not m:
            return None
        return Function(
            name=sanitize(m.group(2)),
            ret_type=m.group(1).strip(),
            params=self._parse_params(m.group(3)),
            is_decl=True,
        )

    # -- function definitions -----------------------------------------------

    def _parse_func_def(self, line: str) -> Optional[Function]:
        m = re.match(r"^define\s+(.+?)\s+(@[\w.]+)\s*\(([^)]*)\)", line)
        if not m:
            return None
        func = Function(
            name=sanitize(m.group(2)),
            ret_type=m.group(1).strip(),
            params=self._parse_params(m.group(3)),
        )
        # Opening brace may be on the same line or the next
        if "{" not in line:
            nxt = self._next()
            if not nxt or "{" not in nxt:
                return func

        cur_block: Optional[Block] = None

        while True:
            line = self._next()
            if line is None:
                break
            if line == "}":
                if cur_block is not None:
                    func.blocks.append(cur_block)
                break

            # Some instructions (e.g. switch) span multiple lines because the
            # square-bracket case list may be wrapped.  Accumulate continuation
            # lines until square brackets are balanced.
            line = self._join_continuation(line)

            # Basic block label: "name:"  (with optional trailing comment)
            lbl = re.match(r"^([\w.$]+):\s*(?:;.*)?$", line)
            if lbl:
                if cur_block is not None:
                    func.blocks.append(cur_block)
                cur_block = Block(label=lbl.group(1))
                continue

            # First instruction before any explicit label → implicit entry block
            if cur_block is None:
                cur_block = Block(label="entry")

            instr = self._parse_instr(line)
            if instr:
                cur_block.instrs.append(instr)

        return func

    def _join_continuation(self, line: str) -> str:
        """If *line* has unbalanced ``[`` brackets, keep reading raw source
        lines (including blank/comment lines) and appending them until the
        brackets are balanced."""
        depth = line.count("[") - line.count("]")
        while depth > 0:
            if self._pos >= len(self._lines):
                break
            cont = self._lines[self._pos]
            self._pos += 1
            stripped = cont.strip()
            if not stripped or stripped.startswith(";"):
                continue
            line = line + " " + stripped
            depth += stripped.count("[") - stripped.count("]")
        return line

    # -- parameters ---------------------------------------------------------

    def _parse_params(self, s: str) -> List[Tuple[str, str]]:
        params: List[Tuple[str, str]] = []
        for i, part in enumerate(split_comma(s)):
            part = strip_param_attrs(part).strip()
            if part in ("...", ""):
                continue
            toks = part.rsplit(None, 1)
            if len(toks) == 2 and toks[1].startswith("%"):
                ty, name = toks[0].strip(), sanitize(toks[1])
            else:
                ty, name = part.strip(), f"arg{i}"
            name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            if name and name[0].isdigit():
                name = "_" + name
            params.append((ty, name))
        return params

    # -- instructions -------------------------------------------------------

    def _parse_instr(self, line: str) -> Optional[Instr]:
        # Strip trailing comment (heuristic; won't handle ';' inside strings)
        line = re.sub(r"\s*;.*$", "", line).strip()
        if not line:
            return None
        result: Optional[str] = None
        m = re.match(r"^(%[\w.]+)\s*=\s*(.+)$", line)
        if m:
            result = sanitize(m.group(1))
            rest = m.group(2).strip()
        else:
            rest = line
        toks = rest.split(None, 1)
        opcode = toks[0]
        raw_rest = toks[1].strip() if len(toks) > 1 else ""
        return Instr(result=result, opcode=opcode, raw_rest=raw_rest)


# ---------------------------------------------------------------------------
# Code generator
# ---------------------------------------------------------------------------

# icmp condition → C operator
_ICMP_OPS: Dict[str, str] = {
    "eq": "==", "ne": "!=",
    "slt": "<",  "sgt": ">",  "sle": "<=", "sge": ">=",
    "ult": "<",  "ugt": ">",  "ule": "<=", "uge": ">=",
}
_UNSIGNED_ICMP = {"ult", "ugt", "ule", "uge"}

# fcmp condition → C operator (approximate; NaN-awareness omitted)
_FCMP_OPS: Dict[str, str] = {
    "oeq": "==", "one": "!=", "olt": "<", "ogt": ">", "ole": "<=", "oge": ">=",
    "ueq": "==", "une": "!=", "ult": "<", "ugt": ">", "ule": "<=", "uge": ">=",
    "true": "1", "false": "0",
}

# Binary arithmetic / bitwise / shift opcodes → C operator
_ARITH_OPS: Dict[str, str] = {
    "add": "+",  "sub": "-",  "mul": "*",
    "fadd": "+", "fsub": "-", "fmul": "*", "fdiv": "/",
    "and": "&", "or": "|", "xor": "^",
    "shl": "<<", "lshr": ">>", "ashr": ">>",
}

_SIGNED_DIV_REM = {"sdiv": "/", "srem": "%"}
_UNSIGNED_DIV_REM = {"udiv": "/", "urem": "%"}

_CAST_OPS = frozenset({
    "trunc", "zext", "sext", "fpext", "fptrunc",
    "fptoui", "fptosi", "uitofp", "sitofp",
    "bitcast", "inttoptr", "ptrtoint", "addrspacecast",
})

# Type alias for the phi-node map used throughout code generation
_PhiMap = Dict[str, List[Tuple[str, str, List[Tuple[str, str]]]]]


class CodeGen:
    """Emit C source code from a parsed :class:`Module`."""

    HEADER = """\
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
"""

    def __init__(self, mod: Module) -> None:
        self._mod = mod
        self._lines: List[str] = []

    def generate(self) -> str:
        self._lines = [self.HEADER]

        # Typedef any named struct types
        for name, body in self._mod.type_defs.items():
            c_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            c_body = llvm_type_to_c(body)
            self._lines.append(f"typedef {c_body} {c_name};")
        if self._mod.type_defs:
            self._lines.append("")

        # Global variables
        for gv in self._mod.globals:
            c_ty = llvm_type_to_c(gv.ty)
            qual = "const " if gv.is_const else ""
            decl = c_decl(c_ty, gv.name)
            if gv.init and gv.init not in ("undef", "poison", "zeroinitializer"):
                self._lines.append(f"{qual}{decl} = {llvm_val(gv.init)};")
            else:
                self._lines.append(f"{qual}{decl};")
        if self._mod.globals:
            self._lines.append("")

        # Forward declarations for all functions (so order doesn't matter)
        for func in self._mod.functions:
            self._lines.append(self._func_sig(func) + ";")
        self._lines.append("")

        # Function bodies
        for func in self._mod.functions:
            if not func.is_decl:
                self._emit_function(func)

        return "\n".join(self._lines) + "\n"

    # -- signature ----------------------------------------------------------

    def _func_sig(self, func: Function) -> str:
        c_ret = llvm_type_to_c(func.ret_type)
        param_strs = [c_decl(llvm_type_to_c(ty), nm) for ty, nm in func.params]
        params_c = ", ".join(param_strs) if param_strs else "void"
        return f"{c_ret} {func.name}({params_c})"

    # -- function body ------------------------------------------------------

    def _emit_function(self, func: Function) -> None:
        self._lines.append(f"{self._func_sig(func)} {{")

        # Collect phi nodes: {block_label: [(result, llvm_type, [(val, pred)])]}
        phi_map: _PhiMap = {}
        for blk in func.blocks:
            phis = []
            for instr in blk.instrs:
                if instr.opcode == "phi" and instr.result:
                    ty, entries = self._parse_phi(instr.raw_rest)
                    phis.append((instr.result, ty, entries))
            if phis:
                phi_map[blk.label] = phis

        # Declare all phi variables at the top so every block can write them
        if phi_map:
            self._lines.append("    /* phi variables */")
            for phis in phi_map.values():
                for res, ty, _entries in phis:
                    self._lines.append(f"    {c_decl(llvm_type_to_c(ty), res)};")
            self._lines.append("")

        for blk in func.blocks:
            self._emit_block(blk, func, phi_map)

        self._lines.append("}")
        self._lines.append("")

    # -- basic block --------------------------------------------------------

    def _emit_block(self, blk: Block, func: Function, phi_map: _PhiMap) -> None:
        # Label (use ``;`` to make it a statement so local declarations can follow)
        self._lines.append(f"  {blk.label}:;")
        for instr in blk.instrs:
            if instr.opcode == "phi":
                continue   # handled via pre-branch assignments
            line = self._emit_instr(instr, blk, phi_map)
            if line:
                # Multi-line statements (e.g. switch) are already indented inside
                for sub in line.split("\n"):
                    self._lines.append(f"    {sub}")

    # -- instruction dispatch -----------------------------------------------

    def _emit_instr(
        self,
        instr: Instr,
        blk: Block,
        phi_map: _PhiMap,
    ) -> Optional[str]:
        op = instr.opcode
        rest = instr.raw_rest
        res = instr.result

        # --- alloca -------------------------------------------------------
        if op == "alloca":
            if res is None:
                return None
            ty = self._extract_alloca_type(rest)
            c_ty = llvm_type_to_c(ty)
            return c_decl(c_ty + "*", res) + "; /* alloca */"

        # --- load ---------------------------------------------------------
        if op == "load":
            if res is None:
                return None
            ty, ptr_val = self._parse_load(rest)
            c_ty = llvm_type_to_c(ty)
            return f"{c_decl(c_ty, res)} = *({c_ty}*){llvm_val(ptr_val)};"

        # --- store --------------------------------------------------------
        if op == "store":
            val, ptr = self._parse_store(rest)
            return f"*{llvm_val(ptr)} = {llvm_val(val)};"

        # --- binary arithmetic / bitwise / shift --------------------------
        if op in _ARITH_OPS:
            ty, a, b = self._parse_binop(rest)
            c_ty = llvm_type_to_c(ty)
            c_op = _ARITH_OPS[op]
            if res is None:
                return None
            return f"{c_decl(c_ty, res)} = {llvm_val(a)} {c_op} {llvm_val(b)};"

        if op in _SIGNED_DIV_REM:
            ty, a, b = self._parse_binop(rest)
            c_ty = llvm_type_to_c(ty)
            c_op = _SIGNED_DIV_REM[op]
            if res is None:
                return None
            return f"{c_decl(c_ty, res)} = {llvm_val(a)} {c_op} {llvm_val(b)};"

        if op in _UNSIGNED_DIV_REM:
            ty, a, b = self._parse_binop(rest)
            c_ty = llvm_type_to_c(ty)
            c_op = _UNSIGNED_DIV_REM[op]
            ucast = f"(unsigned {c_ty})"
            if res is None:
                return None
            return (
                f"{c_decl(c_ty, res)} = "
                f"({c_ty})({ucast}{llvm_val(a)} {c_op} {ucast}{llvm_val(b)});"
            )

        # --- icmp ---------------------------------------------------------
        if op == "icmp":
            if res is None:
                return None
            cond, ty, a, b = self._parse_cmp(rest)
            c_ty = llvm_type_to_c(ty)
            c_op = _ICMP_OPS.get(cond, "==")
            la, lb = llvm_val(a), llvm_val(b)
            if cond in _UNSIGNED_ICMP:
                la = f"(unsigned {c_ty}){la}"
                lb = f"(unsigned {c_ty}){lb}"
            return f"bool {res} = {la} {c_op} {lb};"

        # --- fcmp ---------------------------------------------------------
        if op == "fcmp":
            if res is None:
                return None
            cond, _ty, a, b = self._parse_cmp(rest)
            la, lb = llvm_val(a), llvm_val(b)
            if cond == "true":
                return f"bool {res} = 1;"
            if cond == "false":
                return f"bool {res} = 0;"
            if cond == "ord":
                return f"bool {res} = ({la} == {la}) && ({lb} == {lb});"
            if cond == "uno":
                return f"bool {res} = ({la} != {la}) || ({lb} != {lb});"
            c_op = _FCMP_OPS.get(cond, "==")
            return f"bool {res} = {la} {c_op} {lb};"

        # --- type conversions / casts -------------------------------------
        if op in _CAST_OPS:
            if res is None:
                return None
            src_val, dst_ty = self._parse_cast(rest)
            c_ty = llvm_type_to_c(dst_ty)
            return f"{c_decl(c_ty, res)} = ({c_ty}){llvm_val(src_val)};"

        # --- select -------------------------------------------------------
        if op == "select":
            if res is None:
                return None
            cond_v, true_v, false_v = self._parse_select(rest)
            ty = self._guess_type_from_select(rest)
            c_ty = llvm_type_to_c(ty)
            return (
                f"{c_decl(c_ty, res)} = "
                f"{llvm_val(cond_v)} ? {llvm_val(true_v)} : {llvm_val(false_v)};"
            )

        # --- getelementptr ------------------------------------------------
        if op in ("getelementptr", "getelementptr inbounds"):
            return self._emit_gep(instr)

        # --- call (including tail variants) -------------------------------
        if op in ("call", "tail", "musttail", "notail"):
            return self._emit_call(instr)

        # --- terminators --------------------------------------------------
        if op == "ret":
            if not rest or rest == "void":
                return "return;"
            parts = rest.split(None, 1)
            val = parts[1].strip() if len(parts) > 1 else "0"
            return f"return {llvm_val(val)};"

        if op == "br":
            return self._emit_br(rest, blk, phi_map)

        if op == "switch":
            return self._emit_switch(rest, blk, phi_map)

        if op == "unreachable":
            return "__builtin_unreachable(); /* unreachable */"

        if op == "fence":
            return "__atomic_thread_fence(__ATOMIC_SEQ_CST); /* fence */"

        if op in ("extractvalue", "insertvalue"):
            return (f"void* {res} = 0; /* {op} */" if res
                    else f"/* {op} */")

        # --- fallback ------------------------------------------------------
        if res:
            return f"void* {res} = 0; /* TODO: {op} */"
        return f"/* {op} */"

    # -- phi ----------------------------------------------------------------

    def _parse_phi(
        self, rest: str
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """Parse ``T [ %val, %pred ], ...`` → ``(type, [(val, pred_label)])``."""
        m = re.match(r"^(\S+)\s+\[(.+)$", rest)
        if not m:
            return "i32", []
        ty = m.group(1)
        tail = "[" + m.group(2)
        entries: List[Tuple[str, str]] = [
            (bracket.group(1).strip(), sanitize(bracket.group(2).strip()))
            for bracket in re.finditer(r"\[\s*([^,\]]+),\s*([^\]]+)\s*\]", tail)
        ]
        return ty, entries

    def _phi_assigns(
        self,
        target: str,
        pred: str,
        phi_map: _PhiMap,
    ) -> List[str]:
        """Return assignment statements for phi vars of *target* from *pred*."""
        assignments: List[str] = []
        for res, _ty, entries in phi_map.get(target, []):
            for val, p in entries:
                if p == pred:
                    assignments.append(f"{res} = {llvm_val(val)};")
        return assignments

    # -- branch / switch ----------------------------------------------------

    def _emit_br(
        self, rest: str, blk: Block, phi_map: _PhiMap
    ) -> str:
        parts = [p.strip() for p in rest.split(",")]
        if len(parts) == 1:
            # unconditional br
            dest = sanitize(parts[0].split()[-1])
            stmts = self._phi_assigns(dest, blk.label, phi_map)
            stmts.append(f"goto {dest};")
            return "\n".join(stmts)
        if len(parts) == 3:
            # conditional br i1 %cond, label %T, label %F
            cond = llvm_val(parts[0].split()[-1])
            true_dest  = sanitize(parts[1].split()[-1])
            false_dest = sanitize(parts[2].split()[-1])
            true_stmts  = self._phi_assigns(true_dest,  blk.label, phi_map)
            false_stmts = self._phi_assigns(false_dest, blk.label, phi_map)
            true_body  = " ".join(true_stmts)  + f" goto {true_dest};"
            false_body = " ".join(false_stmts) + f" goto {false_dest};"
            return f"if ({cond}) {{ {true_body} }} else {{ {false_body} }}"
        return f"/* br {rest} */"

    def _emit_switch(
        self, rest: str, blk: Block, phi_map: _PhiMap
    ) -> str:
        m = re.match(
            r"(\S+)\s+(%[\w.]+|\d+),\s*label\s+(%[\w.]+)\s*\[(.+)\]",
            rest,
            re.DOTALL,
        )
        if not m:
            return f"/* switch {rest} */"
        val = llvm_val(m.group(2))
        default = sanitize(m.group(3))
        cases_str = m.group(4)

        lines = [f"switch ({val}) {{"]
        for cm in re.finditer(r"\S+\s+(\S+),\s*label\s+(%[\w.]+)", cases_str):
            case_val   = cm.group(1)
            case_label = sanitize(cm.group(2))
            phi_pre = " ".join(self._phi_assigns(case_label, blk.label, phi_map))
            lines.append(f"  case {case_val}: {phi_pre} goto {case_label};")
        phi_pre = " ".join(self._phi_assigns(default, blk.label, phi_map))
        lines.append(f"  default: {phi_pre} goto {default};")
        lines.append("}")
        return "\n".join(lines)

    # -- call ---------------------------------------------------------------

    def _emit_call(self, instr: Instr) -> str:
        rest = instr.raw_rest
        # For "tail call …", "musttail call …" etc. the opcode captured the
        # first word; strip the leading "call" from raw_rest.
        if instr.opcode in ("tail", "musttail", "notail"):
            rest = re.sub(r"^call\s+", "", rest)
        # Strip calling-convention tokens
        rest = re.sub(
            r"\b(?:fastcc|coldcc|webkit_jscc|anyregcc|preserve_mostcc|"
            r"preserve_allcc|swiftcc|ccc|cc\s+\d+)\b\s*",
            "",
            rest,
        )
        m = re.search(r"(@[\w.]+|%[\w.]+)\s*\(([^)]*)\)", rest)
        if not m:
            return (f"void* {instr.result} = 0; /* call */"
                    if instr.result else "/* call */;")

        func_name = sanitize(m.group(1))
        args_str  = m.group(2).strip()
        c_args: List[str] = []
        for arg in split_comma(args_str):
            arg = strip_param_attrs(arg).strip()
            if not arg:
                continue
            toks = arg.rsplit(None, 1)
            c_args.append(llvm_val(toks[-1] if toks else arg))

        call_expr = f"{func_name}({', '.join(c_args)})"

        res = instr.result
        if res:
            before = rest[: m.start()].strip()
            ret_ty_token = before.split()[-1] if before.split() else "void*"
            c_ret = llvm_type_to_c(ret_ty_token)
            if c_ret == "void":
                return f"{call_expr}; /* void return discarded */"
            return f"{c_decl(c_ret, res)} = {call_expr};"
        return f"{call_expr};"

    # -- getelementptr ------------------------------------------------------

    def _emit_gep(self, instr: Instr) -> Optional[str]:
        rest = instr.raw_rest
        rest = re.sub(r"^inbounds\s+", "", rest)
        parts = split_comma(rest)
        if len(parts) < 2:
            return (f"void* {instr.result} = 0; /* gep */"
                    if instr.result else None)
        base_ty   = parts[0].strip()
        ptr_toks  = parts[1].strip().split()
        ptr_val   = ptr_toks[-1] if ptr_toks else "0"
        indices   = [llvm_val(p.strip().split()[-1]) for p in parts[2:]]
        c_elem    = llvm_type_to_c(base_ty)

        if not indices:
            return (f"void* {instr.result} = (void*){llvm_val(ptr_val)}; /* gep */"
                    if instr.result else None)

        expr = f"({c_elem}*){llvm_val(ptr_val)}"
        for idx in indices:
            expr = f"({expr} + {idx})"

        return (f"void* {instr.result} = (void*){expr};"
                if instr.result else None)

    # -- parsing helpers ----------------------------------------------------

    def _extract_alloca_type(self, rest: str) -> str:
        rest = re.sub(r",\s*align\s+\d+\s*$", "", rest).strip()
        return split_comma(rest)[0].strip() if split_comma(rest) else "i8"

    def _parse_load(self, rest: str) -> Tuple[str, str]:
        rest = re.sub(r"\bvolatile\b\s*", "", rest)
        rest = re.sub(r",\s*align\s+\d+\s*$", "", rest).strip()
        parts = split_comma(rest)
        ty = parts[0].strip()
        ptr_toks = (parts[1].strip().split() if len(parts) > 1 else ["0"])
        return ty, ptr_toks[-1]

    def _parse_store(self, rest: str) -> Tuple[str, str]:
        rest = re.sub(r"\bvolatile\b\s*", "", rest)
        rest = re.sub(r",\s*align\s+\d+\s*$", "", rest).strip()
        parts = split_comma(rest)
        val_toks = parts[0].strip().split()
        ptr_toks = (parts[1].strip().split() if len(parts) > 1 else ["0"])
        return val_toks[-1] if val_toks else "0", ptr_toks[-1]

    def _parse_binop(self, rest: str) -> Tuple[str, str, str]:
        """Return ``(type, lhs, rhs)`` for a binary op, stripping any flags."""
        rest = re.sub(r"\b(?:nuw|nsw|exact)\b\s*", "", rest).strip()
        parts = split_comma(rest)
        ty_a = parts[0].strip()
        b    = parts[1].strip() if len(parts) > 1 else "0"
        toks = ty_a.split()
        ty   = toks[0] if toks else "i32"
        a    = toks[-1] if len(toks) >= 2 else "0"
        return ty, a, b

    def _parse_cmp(self, rest: str) -> Tuple[str, str, str, str]:
        """Return ``(cond, type, lhs, rhs)``."""
        parts = split_comma(rest)
        lhs_toks = parts[0].split()
        cond = lhs_toks[0] if lhs_toks else "eq"
        ty   = lhs_toks[1] if len(lhs_toks) >= 2 else "i32"
        a    = lhs_toks[-1] if len(lhs_toks) >= 3 else "0"
        b    = parts[1].strip() if len(parts) > 1 else "0"
        return cond, ty, a, b

    def _parse_cast(self, rest: str) -> Tuple[str, str]:
        """Return ``(source_value, dest_type)`` for a cast instruction."""
        m = re.match(r"(.+?)\s+to\s+(\S+)\s*$", rest)
        if m:
            src_toks = m.group(1).strip().split()
            return src_toks[-1] if src_toks else "0", m.group(2).strip()
        return "0", "i32"

    def _parse_select(self, rest: str) -> Tuple[str, str, str]:
        """Return ``(cond_val, true_val, false_val)``."""
        parts = split_comma(rest)
        cond  = (parts[0].split()[-1] if parts else "0")
        true_v  = (parts[1].split()[-1] if len(parts) > 1 else "0")
        false_v = (parts[2].split()[-1] if len(parts) > 2 else "0")
        return cond, true_v, false_v

    def _guess_type_from_select(self, rest: str) -> str:
        parts = split_comma(rest)
        if len(parts) >= 2:
            toks = parts[1].split()
            if toks:
                return toks[0]
        return "i32"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert(src: str) -> str:
    """Convert LLVM IR source text *src* to a C source string."""
    mod = Parser(src).parse()
    return CodeGen(mod).generate()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="llvm2c",
        description="Convert LLVM IR (.ll) to C source code",
    )
    ap.add_argument("input", help="Input .ll file (use '-' for stdin)")
    ap.add_argument("-o", "--output", default="-",
                    help="Output .c file (default: stdout)")
    args = ap.parse_args(argv)

    if args.input == "-":
        src = sys.stdin.read()
    else:
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                src = f.read()
        except OSError as e:
            print(f"llvm2c: error: {e}", file=sys.stderr)
            return 1

    out = convert(src)

    if args.output == "-":
        sys.stdout.write(out)
    else:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out)
        except OSError as e:
            print(f"llvm2c: error: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
