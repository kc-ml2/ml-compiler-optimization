; ModuleID = '/scratch/talbn/classifyapp_code/test/1/24.txt.cpp'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external global i8
@a = global [1000 x i32] zeroinitializer, align 16
@l = global i32 0, align 4
@x = global i32 0, align 4
@.str = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_24.txt.cpp, i8* null }]

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
define i32 @main() #2 {
entry:
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %q = alloca i32, align 4
  %p = alloca i32, align 4
  %n = alloca i32, align 4
  %m = alloca i32, align 4
  %y = alloca i32, align 4
  store i32 0, i32* %retval
  %call = call i32 (i8*, ...) @scanf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i32* %n)
  store i32 0, i32* %p, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc.9, %entry
  %0 = load i32, i32* %p, align 4
  %1 = load i32, i32* %n, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end.11

for.body:                                         ; preds = %for.cond
  store i32 1, i32* %q, align 4
  store i32 0, i32* @l, align 4
  %call1 = call i32 (i8*, ...) @scanf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i32* @x)
  store i32 2, i32* %i, align 4
  br label %for.cond.2

for.cond.2:                                       ; preds = %for.inc, %for.body
  %2 = load i32, i32* %i, align 4
  %3 = load i32, i32* @x, align 4
  %cmp3 = icmp sle i32 %2, %3
  br i1 %cmp3, label %for.body.4, label %for.end

for.body.4:                                       ; preds = %for.cond.2
  %4 = load i32, i32* @x, align 4
  %5 = load i32, i32* %i, align 4
  %rem = srem i32 %4, %5
  %cmp5 = icmp eq i32 %rem, 0
  br i1 %cmp5, label %if.then, label %if.end

if.then:                                          ; preds = %for.body.4
  %6 = load i32, i32* %i, align 4
  %7 = load i32, i32* %q, align 4
  %idxprom = sext i32 %7 to i64
  %arrayidx = getelementptr inbounds [1000 x i32], [1000 x i32]* @a, i32 0, i64 %idxprom
  store i32 %6, i32* %arrayidx, align 4
  %8 = load i32, i32* %q, align 4
  %inc = add nsw i32 %8, 1
  store i32 %inc, i32* %q, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body.4
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %9 = load i32, i32* %i, align 4
  %inc6 = add nsw i32 %9, 1
  store i32 %inc6, i32* %i, align 4
  br label %for.cond.2

for.end:                                          ; preds = %for.cond.2
  %10 = load i32, i32* %q, align 4
  %sub = sub nsw i32 %10, 1
  %11 = load i32, i32* @x, align 4
  %call7 = call i32 @_Z3fffiiii(i32 1, i32 %sub, i32 1, i32 %11)
  %12 = load i32, i32* @l, align 4
  %13 = load i32, i32* @x, align 4
  %call8 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i32 0, i32 0), i32 %12, i32 %13)
  br label %for.inc.9

for.inc.9:                                        ; preds = %for.end
  %14 = load i32, i32* %p, align 4
  %inc10 = add nsw i32 %14, 1
  store i32 %inc10, i32* %p, align 4
  br label %for.cond

for.end.11:                                       ; preds = %for.cond
  %15 = load i32, i32* %retval
  ret i32 %15
}

declare i32 @scanf(i8*, ...) #0

; Function Attrs: uwtable
define i32 @_Z3fffiiii(i32 %k, i32 %w, i32 %t, i32 %x) #2 {
entry:
  %retval = alloca i32, align 4
  %k.addr = alloca i32, align 4
  %w.addr = alloca i32, align 4
  %t.addr = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %y = alloca i32, align 4
  %n = alloca i32, align 4
  %m = alloca i32, align 4
  %p = alloca i32, align 4
  store i32 %k, i32* %k.addr, align 4
  store i32 %w, i32* %w.addr, align 4
  store i32 %t, i32* %t.addr, align 4
  store i32 %x, i32* %x.addr, align 4
  %0 = load i32, i32* %t.addr, align 4
  store i32 %0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load i32, i32* %i, align 4
  %2 = load i32, i32* %w.addr, align 4
  %cmp = icmp sle i32 %1, %2
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %3 = load i32, i32* %x.addr, align 4
  %4 = load i32, i32* %i, align 4
  %idxprom = sext i32 %4 to i64
  %arrayidx = getelementptr inbounds [1000 x i32], [1000 x i32]* @a, i32 0, i64 %idxprom
  %5 = load i32, i32* %arrayidx, align 4
  %rem = srem i32 %3, %5
  %cmp1 = icmp eq i32 %rem, 0
  br i1 %cmp1, label %if.then, label %if.end.11

if.then:                                          ; preds = %for.body
  %6 = load i32, i32* %x.addr, align 4
  %7 = load i32, i32* %i, align 4
  %idxprom2 = sext i32 %7 to i64
  %arrayidx3 = getelementptr inbounds [1000 x i32], [1000 x i32]* @a, i32 0, i64 %idxprom2
  %8 = load i32, i32* %arrayidx3, align 4
  %div = sdiv i32 %6, %8
  store i32 %div, i32* %x.addr, align 4
  %9 = load i32, i32* %x.addr, align 4
  %cmp4 = icmp eq i32 %9, 1
  br i1 %cmp4, label %if.then.5, label %if.end

if.then.5:                                        ; preds = %if.then
  %10 = load i32, i32* @l, align 4
  %inc = add nsw i32 %10, 1
  store i32 %inc, i32* @l, align 4
  br label %if.end

if.end:                                           ; preds = %if.then.5, %if.then
  %11 = load i32, i32* %x.addr, align 4
  %cmp6 = icmp sgt i32 %11, 1
  br i1 %cmp6, label %if.then.7, label %if.end.8

if.then.7:                                        ; preds = %if.end
  %12 = load i32, i32* %k.addr, align 4
  %add = add nsw i32 %12, 1
  %13 = load i32, i32* %w.addr, align 4
  %14 = load i32, i32* %i, align 4
  %15 = load i32, i32* %x.addr, align 4
  %call = call i32 @_Z3fffiiii(i32 %add, i32 %13, i32 %14, i32 %15)
  br label %if.end.8

if.end.8:                                         ; preds = %if.then.7, %if.end
  %16 = load i32, i32* %x.addr, align 4
  %17 = load i32, i32* %i, align 4
  %idxprom9 = sext i32 %17 to i64
  %arrayidx10 = getelementptr inbounds [1000 x i32], [1000 x i32]* @a, i32 0, i64 %idxprom9
  %18 = load i32, i32* %arrayidx10, align 4
  %mul = mul nsw i32 %16, %18
  store i32 %mul, i32* %x.addr, align 4
  br label %if.end.11

if.end.11:                                        ; preds = %if.end.8, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end.11
  %19 = load i32, i32* %i, align 4
  %inc12 = add nsw i32 %19, 1
  store i32 %inc12, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  call void @llvm.trap()
  unreachable

return:                                           ; No predecessors!
  %20 = load i32, i32* %retval
  ret i32 %20
}

declare i32 @printf(i8*, ...) #0

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #3

define internal void @_GLOBAL__sub_I_24.txt.cpp() #0 section ".text.startup" {
entry:
  call void @__cxx_global_var_init()
  ret void
}

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noreturn nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.1 (tags/RELEASE_371/final)"}
