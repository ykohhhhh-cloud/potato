
@D1 = external global i16
@Y1 = external global i1
@D2 = external global i16
@A = global i16 0
@A1 = global i16 0
@A2 = global i16 10
@B = global float 0.000000e+00
@B1 = global float 2.000000e+01
@B3 = global float 0x40361999A0000000

define void @main() {
main:
  store i16 109, i16* @A, align 2
  store i16 99, i16* @A1, align 2
  store float 0x4022333340000000, float* @B1, align 4
  store float 1.150000e+01, float* @B, align 4
  store i16 20, i16* @D1, align 2
  store i1 true, i1* @Y1, align 1
  %A = load i16, i16* @A, align 2
  %A2 = load i16, i16* @A2, align 2
  call void @FC_ADD(i16 %A, i16 %A2, i16* @A1)
  %A1 = load i16, i16* @A, align 2
  %D1 = load i16, i16* @D1, align 2
  call void @FC_ADD(i16 %A1, i16 %D1, i16* @D2)
  ret void

funcEnd:                                          ; No predecessors!
}

define void @FC_ADD(i16 %A, i16 %B, i16* %C) {
FC_ADD:
  %A1 = alloca i16, align 2
  store i16 %A, i16* %A1, align 2
  %B2 = alloca i16, align 2
  store i16 %B, i16* %B2, align 2
  store i16 0, i16* %C, align 2
  %load = load i16, i16* %A1, align 2
  %load3 = load i16, i16* %B2, align 2
  %add = add i16 %load, %load3
  store i16 %add, i16* %C, align 2
  ret void

funcEnd:                                          ; No predecessors!
}
