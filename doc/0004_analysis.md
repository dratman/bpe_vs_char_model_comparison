# 0004 analysis output (py/analyze_0004_nope_pilot.py)
```
Loading corpus word vocab from cache: ../txt_local/corpus_high_quality_uppercase_2026_05_08.txt.wordvocab.pkl
CSV: ../plots/0004_val_bpc.csv

val bpc (selected iters):
  iter      PE    NoPE     gap
     0  6.4939  6.4529 -0.0410
   200  2.9753  3.4192 +0.4439
   400  2.3010  2.8956 +0.5947
   600  2.0863  2.4332 +0.3470
  1000  1.9052  2.1050 +0.1998
  2000  1.7573  1.8604 +0.1030
  3000  1.6783  1.7787 +0.1004
  4000  1.6359  1.7312 +0.0954
  5000  1.5898  1.6802 +0.0903
  6000  1.5597  1.6500 +0.0903
  7000  1.5480  1.6341 +0.0861
  8000  1.5138  1.5956 +0.0818
  9000  1.5018  1.5852 +0.0834
best  PE 1.4917 (iter 9800)  NoPE 1.5766 (iter 9800)  gap +0.0848 bpc (+5.69%)
final PE 1.4917  NoPE 1.5766  (iter 9800)

real-word fraction (same sampling seeds both arms):
  iter      PE    NoPE
  2000   96.9%   95.0%
  4000   98.8%   97.9%
  6000   99.2%   98.7%
  8000   99.4%   98.9%
 10000   99.6%   99.3%

PNG: ../plots/0004_nope_pilot.png

== char_nope_pilot_6L_768_nope_mps_final.samples.txt: 6088 words, 6045 real (99.3%), 43 non-words
   real% by word length: 1:100%(n=251)  2:100%(n=1158)  3:100%(n=1625)  4:100%(n=1027)  5:99%(n=639)  6:98%(n=433)  7:98%(n=370)  8:97%(n=226)  9:97%(n=158)  10+:95%(n=201)
   non-words: letters still on a real spelling before the first wrong letter: 3:3  4:8  5:8  6:10  7:5  8+:9
   non-words fixable by one edit: undouble:1(2%)  double:1(2%)  transpose:0(0%)  other1:30(70%)  none:11(26%)

== char_nope_pilot_6L_768_nope_mps_iter2000.samples.txt: 5846 words, 5552 real (95.0%), 294 non-words
   real% by word length: 1:100%(n=156)  2:100%(n=1038)  3:100%(n=1643)  4:99%(n=848)  5:97%(n=616)  6:90%(n=460)  7:85%(n=376)  8:82%(n=283)  9:74%(n=180)  10+:71%(n=246)
   non-words: letters still on a real spelling before the first wrong letter: 2:2  3:30  4:54  5:86  6:48  7:42  8+:32
   non-words fixable by one edit: undouble:15(5%)  double:10(3%)  transpose:6(2%)  other1:146(50%)  none:117(40%)

== char_nope_pilot_6L_768_nope_mps_iter4000.samples.txt: 6080 words, 5953 real (97.9%), 127 non-words
   real% by word length: 1:100%(n=199)  2:100%(n=1012)  3:100%(n=1800)  4:100%(n=999)  5:99%(n=669)  6:96%(n=480)  7:93%(n=388)  8:91%(n=211)  9:88%(n=147)  10+:80%(n=175)
   non-words: letters still on a real spelling before the first wrong letter: 2:1  3:6  4:24  5:38  6:24  7:14  8+:20
   non-words fixable by one edit: undouble:4(3%)  double:3(2%)  transpose:6(5%)  other1:57(45%)  none:57(45%)

== char_nope_pilot_6L_768_nope_mps_iter6000.samples.txt: 5980 words, 5903 real (98.7%), 77 non-words
   real% by word length: 1:100%(n=208)  2:100%(n=1008)  3:100%(n=1696)  4:100%(n=998)  5:100%(n=624)  6:97%(n=381)  7:96%(n=395)  8:96%(n=272)  9:95%(n=204)  10+:87%(n=194)
   non-words: letters still on a real spelling before the first wrong letter: 3:4  4:12  5:19  6:14  7:10  8+:18
   non-words fixable by one edit: undouble:6(8%)  double:1(1%)  transpose:3(4%)  other1:37(48%)  none:30(39%)

== char_nope_pilot_6L_768_nope_mps_iter8000.samples.txt: 6036 words, 5970 real (98.9%), 66 non-words
   real% by word length: 1:100%(n=215)  2:100%(n=1081)  3:100%(n=1648)  4:100%(n=991)  5:100%(n=636)  6:99%(n=481)  7:96%(n=392)  8:95%(n=241)  9:92%(n=156)  10+:93%(n=195)
   non-words: letters still on a real spelling before the first wrong letter: 3:2  4:17  5:18  6:13  7:7  8+:9
   non-words fixable by one edit: undouble:4(6%)  double:1(2%)  transpose:2(3%)  other1:28(42%)  none:31(47%)

== char_nope_pilot_6L_768_pe_mps_final.samples.txt: 6072 words, 6048 real (99.6%), 24 non-words
   real% by word length: 1:100%(n=223)  2:100%(n=1163)  3:100%(n=1648)  4:100%(n=981)  5:100%(n=626)  6:99%(n=477)  7:99%(n=374)  8:99%(n=230)  9:97%(n=136)  10+:97%(n=214)
   non-words: letters still on a real spelling before the first wrong letter: 3:1  4:5  5:6  6:4  7:2  8+:6
   non-words fixable by one edit: undouble:0(0%)  double:0(0%)  transpose:0(0%)  other1:14(58%)  none:10(42%)

== char_nope_pilot_6L_768_pe_mps_iter2000.samples.txt: 6023 words, 5834 real (96.9%), 189 non-words
   real% by word length: 1:100%(n=218)  2:100%(n=1015)  3:100%(n=1614)  4:100%(n=1056)  5:98%(n=650)  6:93%(n=487)  7:90%(n=402)  8:85%(n=246)  9:85%(n=180)  10+:76%(n=155)
   non-words: letters still on a real spelling before the first wrong letter: 2:2  3:5  4:35  5:64  6:35  7:27  8+:21
   non-words fixable by one edit: undouble:5(3%)  double:3(2%)  transpose:5(3%)  other1:101(53%)  none:75(40%)

== char_nope_pilot_6L_768_pe_mps_iter4000.samples.txt: 6097 words, 6026 real (98.8%), 71 non-words
   real% by word length: 1:100%(n=220)  2:100%(n=1068)  3:100%(n=1705)  4:100%(n=1112)  5:99%(n=644)  6:98%(n=448)  7:97%(n=350)  8:93%(n=230)  9:94%(n=157)  10+:88%(n=163)
   non-words: letters still on a real spelling before the first wrong letter: 3:3  4:16  5:18  6:17  7:9  8+:8
   non-words fixable by one edit: undouble:1(1%)  double:0(0%)  transpose:3(4%)  other1:37(52%)  none:30(42%)

== char_nope_pilot_6L_768_pe_mps_iter6000.samples.txt: 5992 words, 5942 real (99.2%), 50 non-words
   real% by word length: 1:100%(n=224)  2:100%(n=1003)  3:100%(n=1592)  4:100%(n=1062)  5:100%(n=679)  6:98%(n=448)  7:98%(n=361)  8:97%(n=259)  9:94%(n=178)  10+:92%(n=186)
   non-words: letters still on a real spelling before the first wrong letter: 4:10  5:13  6:10  7:3  8+:14
   non-words fixable by one edit: undouble:0(0%)  double:0(0%)  transpose:1(2%)  other1:26(52%)  none:23(46%)

== char_nope_pilot_6L_768_pe_mps_iter8000.samples.txt: 5989 words, 5955 real (99.4%), 34 non-words
   real% by word length: 1:100%(n=207)  2:100%(n=1131)  3:100%(n=1634)  4:100%(n=935)  5:100%(n=612)  6:100%(n=419)  7:98%(n=376)  8:97%(n=298)  9:95%(n=148)  10+:97%(n=229)
   non-words: letters still on a real spelling before the first wrong letter: 4:7  5:12  6:10  7:2  8+:3
   non-words fixable by one edit: undouble:1(3%)  double:0(0%)  transpose:2(6%)  other1:12(35%)  none:19(56%)
```

# 0004 per-character loss breakdown (py/nope_per_char_loss.py)
```
pe: pt/char_nope_pilot_6L_768_pe_mps_final.pt | wpe present: True
nope: pt/char_nope_pilot_6L_768_nope_mps_final.pt | wpe present: False

400 windows x 256 chars = 102,400 predictions from the held-out (val) region; loss in bits per character
bucket                                       n      PE    NoPE  NoPE-PE  ratio
all                                     102400   1.483   1.568   +0.085   1.06
window pos 000-007                        3200   2.275   2.296   +0.021   1.01
window pos 008-031                        9600   1.567   1.628   +0.061   1.04
window pos 032-127                       38400   1.468   1.550   +0.082   1.06
window pos 128-255                       51200   1.430   1.525   +0.095   1.07
letter #0 in word                        18369   3.583   3.686   +0.102   1.03
letter #1 in word                        17741   1.606   1.692   +0.086   1.05
letter #2 in word                        14480   1.416   1.526   +0.110   1.08
letter #3 in word                        10187   0.977   1.072   +0.095   1.10
letter #4 in word                         6995   0.650   0.728   +0.078   1.12
letter #5 in word                         4922   0.471   0.538   +0.068   1.14
letter #6 in word                         3400   0.389   0.467   +0.078   1.20
letter #7 in word                         2117   0.359   0.407   +0.048   1.13
letter #8+ in word                        2485   0.359   0.427   +0.068   1.19
in-word: 2nd of double letter             1839   0.982   1.087   +0.105   1.11
in-word: other letters (pos>=1)          60488   1.107   1.195   +0.088   1.08
char right after word ends               18395   0.791   0.857   +0.066   1.08
```
