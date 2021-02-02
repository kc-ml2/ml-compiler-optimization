; ModuleID = '/scratch/talbn/classifyapp_code/test/1/73.txt.cpp'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%"class.std::basic_istream" = type { i32 (...)**, i64, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", %"class.std::basic_ostream"*, i8, i8, %"class.std::basic_streambuf"*, %"class.std::ctype"*, %"class.std::num_put"*, %"class.std::num_get"* }
%"class.std::ios_base" = type { i32 (...)**, i64, i64, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, %"struct.std::ios_base::_Words"*, %"class.std::locale" }
%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"class.std::ios_base"*, i32)*, i32, i32 }
%"struct.std::ios_base::_Words" = type { i8*, i64 }
%"class.std::locale" = type { %"class.std::locale::_Impl"* }
%"class.std::locale::_Impl" = type { i32, %"class.std::locale::facet"**, i64, %"class.std::locale::facet"**, i8** }
%"class.std::locale::facet" = type <{ i32 (...)**, i32, [4 x i8] }>
%"class.std::basic_ostream" = type { i32 (...)**, %"class.std::basic_ios" }
%"class.std::basic_streambuf" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"class.std::locale" }
%"class.std::ctype" = type <{ %"class.std::locale::facet.base", [4 x i8], %struct.__locale_struct*, i8, [7 x i8], i32*, i32*, i16*, i8, [256 x i8], [256 x i8], i8, [6 x i8] }>
%"class.std::locale::facet.base" = type <{ i32 (...)**, i32 }>
%struct.__locale_struct = type { [13 x %struct.__locale_data*], i16*, i32*, i32*, [13 x i8*] }
%struct.__locale_data = type opaque
%"class.std::num_put" = type { %"class.std::locale::facet.base", [4 x i8] }
%"class.std::num_get" = type { %"class.std::locale::facet.base", [4 x i8] }

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external global i8
@_ZSt3cin = external global %"class.std::basic_istream", align 8
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_73.txt.cpp, i8* null }]

define internal void @__cxx_global_var_init() #0 section ".text.startup" {
entry:
  call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* @_ZStL8__ioinit)
  %0 = call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i32 0, i32 0), i8* @__dso_handle) #1
  ret void
}

declare void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) #0

declare void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) #0

; Function Attrs: nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) #1

; Function Attrs: uwtable
define i32 @_Z3funii(i32 %m, i32 %n) #2 {
entry:
  %m.addr = alloca i32, align 4
  %n.addr = alloca i32, align 4
  %i = alloca i32, align 4
  %sum = alloca i32, align 4
  store i32 %m, i32* %m.addr, align 4
  store i32 %n, i32* %n.addr, align 4
  store i32 1, i32* %sum, align 4
  %0 = load i32, i32* %n.addr, align 4
  %cmp = icmp eq i32 %0, 1
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 2, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %if.then
  %1 = load i32, i32* %i, align 4
  %2 = load i32, i32* %i, align 4
  %mul = mul nsw i32 %1, %2
  %3 = load i32, i32* %m.addr, align 4
  %cmp1 = icmp sle i32 %mul, %3
  br i1 %cmp1, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %4 = load i32, i32* %m.addr, align 4
  %5 = load i32, i32* %i, align 4
  %rem = srem i32 %4, %5
  %cmp2 = icmp eq i32 %rem, 0
  br i1 %cmp2, label %if.then.3, label %if.end

if.then.3:                                        ; preds = %for.body
  %6 = load i32, i32* %m.addr, align 4
  %7 = load i32, i32* %i, align 4
  %div = sdiv i32 %6, %7
  %8 = load i32, i32* %i, align 4
  %call = call i32 @_Z3funii(i32 %div, i32 %8)
  %9 = load i32, i32* %sum, align 4
  %add = add nsw i32 %9, %call
  store i32 %add, i32* %sum, align 4
  br label %if.end

if.end:                                           ; preds = %if.then.3, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %10 = load i32, i32* %i, align 4
  %inc = add nsw i32 %10, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br label %if.end.18

if.else:                                          ; preds = %entry
  %11 = load i32, i32* %n.addr, align 4
  store i32 %11, i32* %i, align 4
  br label %for.cond.4

for.cond.4:                                       ; preds = %for.inc.15, %if.else
  %12 = load i32, i32* %i, align 4
  %13 = load i32, i32* %i, align 4
  %mul5 = mul nsw i32 %12, %13
  %14 = load i32, i32* %m.addr, align 4
  %cmp6 = icmp sle i32 %mul5, %14
  br i1 %cmp6, label %for.body.7, label %for.end.17

for.body.7:                                       ; preds = %for.cond.4
  %15 = load i32, i32* %m.addr, align 4
  %16 = load i32, i32* %i, align 4
  %rem8 = srem i32 %15, %16
  %cmp9 = icmp eq i32 %rem8, 0
  br i1 %cmp9, label %if.then.10, label %if.end.14

if.then.10:                                       ; preds = %for.body.7
  %17 = load i32, i32* %m.addr, align 4
  %18 = load i32, i32* %i, align 4
  %div11 = sdiv i32 %17, %18
  %19 = load i32, i32* %i, align 4
  %call12 = call i32 @_Z3funii(i32 %div11, i32 %19)
  %20 = load i32, i32* %sum, align 4
  %add13 = add nsw i32 %20, %call12
  store i32 %add13, i32* %sum, align 4
  br label %if.end.14

if.end.14:                                        ; preds = %if.then.10, %for.body.7
  br label %for.inc.15

for.inc.15:                                       ; preds = %if.end.14
  %21 = load i32, i32* %i, align 4
  %inc16 = add nsw i32 %21, 1
  store i32 %inc16, i32* %i, align 4
  br label %for.cond.4

for.end.17:                                       ; preds = %for.cond.4
  br label %if.end.18

if.end.18:                                        ; preds = %for.end.17, %for.end
  %22 = load i32, i32* %sum, align 4
  ret i32 %22
}

; Function Attrs: uwtable
define i32 @main() #2 {
entry:
  %retval = alloca i32, align 4
  %n = alloca i32, align 4
  %i = alloca i32, align 4
  %b = alloca i32, align 4
  %a = alloca [1000 x i32], align 16
  store i32 0, i32* %retval
  %call = call dereferenceable(280) %"class.std::basic_istream"* @_ZNSirsERi(%"class.std::basic_istream"* @_ZSt3cin, i32* dereferenceable(4) %n)
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %1 = load i32, i32* %n, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %call1 = call dereferenceable(280) %"class.std::basic_istream"* @_ZNSirsERi(%"class.std::basic_istream"* @_ZSt3cin, i32* dereferenceable(4) %b)
  %2 = load i32, i32* %b, align 4
  %call2 = call i32 @_Z3funii(i32 %2, i32 1)
  %3 = load i32, i32* %i, align 4
  %idxprom = sext i32 %3 to i64
  %arrayidx = getelementptr inbounds [1000 x i32], [1000 x i32]* %a, i32 0, i64 %idxprom
  store i32 %call2, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %4 = load i32, i32* %i, align 4
  %inc = add nsw i32 %4, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  store i32 0, i32* %i, align 4
  br label %for.cond.3

for.cond.3:                                       ; preds = %for.inc.10, %for.end
  %5 = load i32, i32* %i, align 4
  %6 = load i32, i32* %n, align 4
  %cmp4 = icmp slt i32 %5, %6
  br i1 %cmp4, label %for.body.5, label %for.end.12

for.body.5:                                       ; preds = %for.cond.3
  %7 = load i32, i32* %i, align 4
  %idxprom6 = sext i32 %7 to i64
  %arrayidx7 = getelementptr inbounds [1000 x i32], [1000 x i32]* %a, i32 0, i64 %idxprom6
  %8 = load i32, i32* %arrayidx7, align 4
  %call8 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* @_ZSt4cout, i32 %8)
  %call9 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %call8, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  br label %for.inc.10

for.inc.10:                                       ; preds = %for.body.5
  %9 = load i32, i32* %i, align 4
  %inc11 = add nsw i32 %9, 1
  store i32 %inc11, i32* %i, align 4
  br label %for.cond.3

for.end.12:                                       ; preds = %for.cond.3
  %10 = load i32, i32* %retval
  ret i32 %10
}

declare dereferenceable(280) %"class.std::basic_istream"* @_ZNSirsERi(%"class.std::basic_istream"*, i32* dereferenceable(4)) #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"*, i32) #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"*, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)*) #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(%"class.std::basic_ostream"* dereferenceable(272)) #0

define internal void @_GLOBAL__sub_I_73.txt.cpp() #0 section ".text.startup" {
entry:
  call void @__cxx_global_var_init()
  ret void
}

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.1 (tags/RELEASE_371/final)"}
