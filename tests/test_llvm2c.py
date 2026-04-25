"""Tests for llvm2c – LLVM IR to C converter."""

import sys
import os
import textwrap
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llvm2c import (
    convert,
    llvm_type_to_c,
    c_decl,
    sanitize,
    llvm_val,
    split_comma,
    Parser,
    CodeGen,
    Module,
)


# ---------------------------------------------------------------------------
# Unit tests – type helpers
# ---------------------------------------------------------------------------

class TestLlvmTypeToC:
    def test_void(self):
        assert llvm_type_to_c("void") == "void"

    def test_integers(self):
        assert llvm_type_to_c("i1")  == "bool"
        assert llvm_type_to_c("i8")  == "int8_t"
        assert llvm_type_to_c("i16") == "int16_t"
        assert llvm_type_to_c("i32") == "int32_t"
        assert llvm_type_to_c("i64") == "int64_t"

    def test_floats(self):
        assert llvm_type_to_c("float")  == "float"
        assert llvm_type_to_c("double") == "double"
        assert llvm_type_to_c("half")   == "float"

    def test_opaque_pointer(self):
        assert llvm_type_to_c("ptr") == "void*"

    def test_typed_pointer(self):
        assert llvm_type_to_c("i32*") == "int32_t*"
        assert llvm_type_to_c("i8*")  == "int8_t*"

    def test_array(self):
        assert llvm_type_to_c("[4 x i32]") == "int32_t[4]"

    def test_vector(self):
        assert llvm_type_to_c("<4 x float>") == "float[4]"

    def test_named_struct(self):
        assert llvm_type_to_c("%MyStruct") == "MyStruct"


class TestCDecl:
    def test_plain(self):
        assert c_decl("int32_t", "x") == "int32_t x"

    def test_pointer(self):
        assert c_decl("int32_t*", "p") == "int32_t* p"

    def test_array(self):
        assert c_decl("int32_t[10]", "arr") == "int32_t arr[10]"


class TestSanitize:
    def test_sigil_percent(self):
        assert sanitize("%foo") == "foo"

    def test_sigil_at(self):
        assert sanitize("@bar") == "bar"

    def test_numeric_name(self):
        assert sanitize("%1") == "_1"

    def test_dot_in_name(self):
        assert sanitize("%foo.bar") == "foo_bar"


class TestLlvmVal:
    def test_null(self):
        assert llvm_val("null") == "NULL"

    def test_true_false(self):
        assert llvm_val("true")  == "1"
        assert llvm_val("false") == "0"

    def test_undef(self):
        assert llvm_val("undef") == "0 /*undef*/"

    def test_integer(self):
        assert llvm_val("42") == "42"
        assert llvm_val("-7") == "-7"

    def test_percent_value(self):
        assert llvm_val("%x") == "x"

    def test_at_value(self):
        assert llvm_val("@g") == "g"


class TestSplitComma:
    def test_simple(self):
        assert split_comma("a, b, c") == ["a", "b", "c"]

    def test_nested(self):
        assert split_comma("[4 x i32], i32 %x") == ["[4 x i32]", "i32 %x"]

    def test_angle(self):
        assert split_comma("<4 x float>, float %f") == ["<4 x float>", "float %f"]


# ---------------------------------------------------------------------------
# Integration tests – convert() produces correct C snippets
# ---------------------------------------------------------------------------

def _has(output: str, fragment: str) -> bool:
    """Return True if *fragment* appears literally in *output*."""
    return fragment in output


class TestConvertTypeMapping:
    """The generated C file includes correct standard-header types."""

    def test_header_includes(self):
        out = convert("; empty module\n")
        assert "#include <stdint.h>" in out
        assert "#include <stdbool.h>" in out


class TestConvertFunctionDeclaration:
    """Function declarations generate correct C signatures."""

    IR = textwrap.dedent("""\
        declare i32 @printf(ptr %fmt, ...)
    """)

    def test_forward_decl(self):
        out = convert(self.IR)
        assert "int32_t printf" in out


class TestConvertSimpleFunction:
    """A function that adds two i32 values."""

    IR = textwrap.dedent("""\
        define i32 @add(i32 noundef %a, i32 noundef %b) {
          %result = add i32 %a, %b
          ret i32 %result
        }
    """)

    def test_function_exists(self):
        out = convert(self.IR)
        assert "int32_t add(" in out

    def test_addition(self):
        out = convert(self.IR)
        assert "a + b" in out

    def test_return(self):
        out = convert(self.IR)
        assert "return result;" in out


class TestConvertArithmetic:
    """All basic integer arithmetic opcodes."""

    def _out(self, opcode: str, c_op: str) -> str:
        ir = textwrap.dedent(f"""\
            define i32 @f(i32 %a, i32 %b) {{
              %r = {opcode} i32 %a, %b
              ret i32 %r
            }}
        """)
        return convert(ir)

    def test_sub(self):
        assert "a - b" in self._out("sub", "-")

    def test_mul(self):
        assert "a * b" in self._out("mul", "*")

    def test_sdiv(self):
        assert "a / b" in self._out("sdiv", "/")

    def test_srem(self):
        assert "a % b" in self._out("srem", "%")

    def test_udiv(self):
        out = self._out("udiv", "/")
        assert "/" in out

    def test_and(self):
        assert "a & b" in self._out("and", "&")

    def test_or(self):
        assert "a | b" in self._out("or", "|")

    def test_xor(self):
        assert "a ^ b" in self._out("xor", "^")

    def test_shl(self):
        assert "a << b" in self._out("shl", "<<")

    def test_lshr(self):
        assert "a >> b" in self._out("lshr", ">>")


class TestConvertFloatArithmetic:
    """Basic floating-point opcodes."""

    def _out(self, opcode: str) -> str:
        ir = textwrap.dedent(f"""\
            define double @f(double %a, double %b) {{
              %r = {opcode} double %a, %b
              ret double %r
            }}
        """)
        return convert(ir)

    def test_fadd(self):
        assert "a + b" in self._out("fadd")

    def test_fsub(self):
        assert "a - b" in self._out("fsub")

    def test_fmul(self):
        assert "a * b" in self._out("fmul")

    def test_fdiv(self):
        assert "a / b" in self._out("fdiv")


class TestConvertIcmp:
    """Integer comparison generates correct C predicates."""

    def _out(self, cond: str) -> str:
        ir = textwrap.dedent(f"""\
            define i1 @cmp(i32 %a, i32 %b) {{
              %r = icmp {cond} i32 %a, %b
              ret i1 %r
            }}
        """)
        return convert(ir)

    def test_eq(self):
        assert "a == b" in self._out("eq")

    def test_ne(self):
        assert "a != b" in self._out("ne")

    def test_slt(self):
        assert "a < b" in self._out("slt")

    def test_sgt(self):
        assert "a > b" in self._out("sgt")

    def test_sle(self):
        assert "a <= b" in self._out("sle")

    def test_sge(self):
        assert "a >= b" in self._out("sge")

    def test_ult_unsigned_cast(self):
        out = self._out("ult")
        assert "unsigned" in out
        assert "<" in out


class TestConvertFcmp:
    def test_oeq(self):
        ir = textwrap.dedent("""\
            define i1 @fcmp_test(double %a, double %b) {
              %r = fcmp oeq double %a, %b
              ret i1 %r
            }
        """)
        out = convert(ir)
        assert "a == b" in out

    def test_ord(self):
        ir = textwrap.dedent("""\
            define i1 @f(double %a, double %b) {
              %r = fcmp ord double %a, %b
              ret i1 %r
            }
        """)
        out = convert(ir)
        assert "==" in out   # NaN check  a == a


class TestConvertMemory:
    """alloca / load / store generate sensible C."""

    IR = textwrap.dedent("""\
        define i32 @mem_test(i32 %x) {
          %p = alloca i32
          store i32 %x, ptr %p
          %v = load i32, ptr %p
          ret i32 %v
        }
    """)

    def test_alloca(self):
        out = convert(self.IR)
        assert "alloca" in out   # comment or variable

    def test_store(self):
        out = convert(self.IR)
        assert "*p = x;" in out

    def test_load(self):
        out = convert(self.IR)
        assert "int32_t v =" in out


class TestConvertCasts:
    """Type-cast instructions."""

    def _out(self, op: str, src_ty: str, dst_ty: str) -> str:
        ir = textwrap.dedent(f"""\
            define {dst_ty} @f({src_ty} %x) {{
              %r = {op} {src_ty} %x to {dst_ty}
              ret {dst_ty} %r
            }}
        """)
        return convert(ir)

    def test_trunc(self):
        out = self._out("trunc", "i32", "i8")
        assert "(int8_t)" in out

    def test_zext(self):
        out = self._out("zext", "i8", "i32")
        assert "(int32_t)" in out

    def test_sext(self):
        out = self._out("sext", "i8", "i64")
        assert "(int64_t)" in out

    def test_bitcast(self):
        out = self._out("bitcast", "i32", "float")
        assert "(float)" in out

    def test_fpext(self):
        out = self._out("fpext", "float", "double")
        assert "(double)" in out


class TestConvertSelect:
    IR = textwrap.dedent("""\
        define i32 @sel(i1 %cond, i32 %a, i32 %b) {
          %r = select i1 %cond, i32 %a, i32 %b
          ret i32 %r
        }
    """)

    def test_ternary(self):
        out = convert(self.IR)
        assert "cond ? a : b" in out


class TestConvertBranch:
    """Conditional and unconditional branches."""

    UNCOND_IR = textwrap.dedent("""\
        define void @f() {
        entry:
          br label %done
        done:
          ret void
        }
    """)

    COND_IR = textwrap.dedent("""\
        define void @f(i1 %c) {
        entry:
          br i1 %c, label %yes, label %no
        yes:
          ret void
        no:
          ret void
        }
    """)

    def test_unconditional_goto(self):
        out = convert(self.UNCOND_IR)
        assert "goto done" in out

    def test_conditional_if(self):
        out = convert(self.COND_IR)
        assert "if (c)" in out
        assert "goto yes" in out
        assert "goto no" in out


class TestConvertPhi:
    """PHI nodes are lowered to mutable variables."""

    IR = textwrap.dedent("""\
        define i32 @countdown(i32 %n) {
        entry:
          br label %loop
        loop:
          %i = phi i32 [ %n, %entry ], [ %next, %loop ]
          %next = sub i32 %i, 1
          %done = icmp eq i32 %i, 0
          br i1 %done, label %exit, label %loop
        exit:
          ret i32 %i
        }
    """)

    def test_phi_var_declared(self):
        out = convert(self.IR)
        assert "int32_t i;" in out

    def test_phi_entry_assign(self):
        out = convert(self.IR)
        # Before jumping to %loop from %entry, i = n
        assert "i = n" in out

    def test_phi_back_edge_assign(self):
        out = convert(self.IR)
        # Before looping back, i = next
        assert "i = next" in out


class TestConvertCall:
    """Function calls."""

    IR = textwrap.dedent("""\
        declare i32 @abs(i32)
        define i32 @call_abs(i32 %x) {
          %r = call i32 @abs(i32 %x)
          ret i32 %r
        }
    """)

    def test_call_expr(self):
        out = convert(self.IR)
        assert "abs(x)" in out

    def test_call_result_assigned(self):
        out = convert(self.IR)
        assert "int32_t r = abs(x);" in out


class TestConvertGlobalVar:
    """Global variables."""

    IR = textwrap.dedent("""\
        @g = global i32 42
        @c = constant i32 7
    """)

    def test_global_var(self):
        out = convert(self.IR)
        assert "int32_t g = 42;" in out

    def test_const_var(self):
        out = convert(self.IR)
        assert "const int32_t c = 7;" in out


class TestConvertSwitch:
    """switch statements produce a C switch."""

    IR = textwrap.dedent("""\
        define void @sw(i32 %x) {
        entry:
          switch i32 %x, label %default [ i32 1, label %case1
                                          i32 2, label %case2 ]
        case1:
          ret void
        case2:
          ret void
        default:
          ret void
        }
    """)

    def test_switch_keyword(self):
        out = convert(self.IR)
        assert "switch (x)" in out

    def test_case_labels(self):
        out = convert(self.IR)
        assert "case 1:" in out
        assert "case 2:" in out


class TestConvertGEP:
    """getelementptr produces a pointer-arithmetic expression."""

    IR = textwrap.dedent("""\
        define ptr @gep(ptr %arr, i32 %idx) {
          %p = getelementptr i32, ptr %arr, i32 %idx
          ret ptr %p
        }
    """)

    def test_gep_present(self):
        out = convert(self.IR)
        assert "arr" in out
        assert "idx" in out


class TestConvertNSWFlags:
    """Arithmetic with 'nuw' / 'nsw' flags still generates simple C."""

    IR = textwrap.dedent("""\
        define i32 @f(i32 %a, i32 %b) {
          %r = add nuw nsw i32 %a, %b
          ret i32 %r
        }
    """)

    def test_add_flags_stripped(self):
        out = convert(self.IR)
        assert "a + b" in out


class TestConvertUnreachable:
    IR = textwrap.dedent("""\
        define void @f() {
          unreachable
        }
    """)

    def test_unreachable(self):
        out = convert(self.IR)
        assert "__builtin_unreachable()" in out


class TestConvertVoidReturn:
    IR = textwrap.dedent("""\
        define void @noop() {
          ret void
        }
    """)

    def test_void_return(self):
        out = convert(self.IR)
        assert "return;" in out


class TestConvertMultipleBlocks:
    """A function with multiple explicit basic blocks."""

    IR = textwrap.dedent("""\
        define i32 @abs_val(i32 %x) {
        entry:
          %neg = icmp slt i32 %x, 0
          br i1 %neg, label %negate, label %done
        negate:
          %r = sub i32 0, %x
          br label %done
        done:
          %result = phi i32 [ %x, %entry ], [ %r, %negate ]
          ret i32 %result
        }
    """)

    def test_labels_present(self):
        out = convert(self.IR)
        assert "negate:" in out
        assert "done:" in out

    def test_phi_declared(self):
        out = convert(self.IR)
        assert "int32_t result;" in out


class TestCLI:
    """CLI integration tests using main()."""

    def test_stdin_stdout(self, tmp_path, capsys):
        import llvm2c
        ir_file = tmp_path / "test.ll"
        ir_file.write_text(textwrap.dedent("""\
            define i32 @answer() {
              ret i32 42
            }
        """))
        ret = llvm2c.main([str(ir_file)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "int32_t answer" in out

    def test_output_file(self, tmp_path):
        import llvm2c
        ir_file  = tmp_path / "test.ll"
        out_file = tmp_path / "test.c"
        ir_file.write_text("define void @f() { ret void }\n")
        ret = llvm2c.main([str(ir_file), "-o", str(out_file)])
        assert ret == 0
        content = out_file.read_text()
        assert "void f(" in content

    def test_missing_file(self, capsys):
        import llvm2c
        ret = llvm2c.main(["/nonexistent/file.ll"])
        assert ret == 1
        err = capsys.readouterr().err
        assert "error" in err.lower()
