$include = (Get-Item ..\..\tools\fasm\INCLUDE).FullName
$env:INCLUDE=$include
..\..\tools\fasm\fasm.exe .\equihash-xenon\Windows\test_avx1.asm .\test_avx1.exe
..\..\tools\fasm\fasm.exe .\equihash-xenon\Windows\test_avx2.asm .\test_avx2.exe
..\..\tools\fasm\fasm.exe .\equihash-xenon\Windows\quickbench_avx1.asm .\quickbench_avx1.exe
..\..\tools\fasm\fasm.exe .\equihash-xenon\Windows\quickbench_avx2.asm .\quickbench_avx2.exe
..\..\tools\fasm\fasm.exe .\equihash-xenon\Windows\equihash_avx1.asm .\equihash_avx1.obj
..\..\tools\fasm\fasm.exe .\equihash-xenon\Windows\equihash_avx2.asm .\equihash_avx2.obj