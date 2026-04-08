# Diary Entry 070: Depth Comparison via Functional Summary
## 3L, 4L, 6L, 7L models on "he said" — position 5 (i→d)

### Side-by-side: the pipeline stretches with depth

**3-layer model** (0.60M params, val=1.238):
```
L0: diffuse attn, MLP quiet          lens: i=99%  vis=24%
L1: H1→s(53%) H2→a(66%), MLP wakes   lens: i=54%  vis=19%
L2: H0→s(92%) H1→s(100%) H3→a(91%)  lens: l=41%  vis=58%  push=+1.08
```
Word reading and prediction crammed into L1+L2. No broadcasting layer.
Final P(d)=39%.

**4-layer model** (0.80M params, val=1.231):
```
L0: diffuse                           lens: i=96%  vis=27%
L1: some pos0 reading, ATTN-dom       lens: i=70%  vis=25%
L2: H0→a(84%) H2→s(78%) H3→␣(60%)   lens: s=45%  vis=28%
L3: H0→s(82%) H2→a(94%) H3→s(85%)   lens: d=45%  vis=63%
```
Dedicated broadcasting emerges at L1. Word reading at L2-L3.
Final P(d)=45%.

**6-layer model** (1.19M params, val=1.185):
```
L0: diffuse                           lens: i=99%  vis=48%
L1: ALL pos0 (25-48%), ATTN-dom       lens: i=77%  vis=39%
L2: H2→e(52%) H3→pos0(42%)           lens: i=42%  vis=32%
L3: H2→s(80%) H3→a(84%), MLP wakes   lens: n=58%  vis=33%
L4: H0→s(80%) H1→a(73%) H2→s(98%)   lens: d=19%  vis=34%
L5: mixed, MLP push=+1.14            lens: d=70%  vis=61%
```
Full pipeline: broadcast→cross-word→word-read→predict.
Final P(d)=70%.

**7-layer model** (1.39M params, val=1.164):
```
L0: diffuse                           lens: i=99%  vis=51%
L1: H1→h(51%), some pos0              lens: i=74%  vis=32%
L2: pos0 reading (28-42%)             lens: i=32%  vis=30%
L3: diffuse, MLP push=+0.38          lens: d=27%  vis=30%
L4: H0→s(99%) H1→i(84%) H2→a(97%)   lens: d=47%  vis=29%
L5: H0→␣(58%) H3→␣(57%), SUPPRESSOR  lens: d=38%  vis=33%
L6: ALL→h (55-82%), MLP push=+1.72   lens: d=82%  vis=74%
```
Pipeline + dedicated suppressor at L5 + massive final push at L6.
Final P(d)=82%.

### Key patterns visible at a glance

1. **L0 is always diffuse.** 3L, 4L, 6L, 7L — no exception.

2. **Broadcasting appears at 4+ layers.** 3L has no broadcasting.
   4L has partial. 6L has full (all 4 heads FST at L1). 7L has full.

3. **Word reading always uses the same attention pattern.** Every model
   reads 's' and 'a' from earlier positions. The specific layer differs
   (L1-2 for 3L, L2-3 for 4L, L3-4 for 6L, L4 for 7L) but the WHAT
   is identical.

4. **The final layer always has the biggest MLP push.** 3L: +1.08.
   4L: +0.98. 6L: +1.14. 7L: +1.72. The final layer carries the
   prediction regardless of depth.

5. **The suppressor only appears at 7 layers.** L5 in the 7L model
   pushes -0.13. No other model has a negative push. You need enough
   depth for the model to "afford" a calibration layer.

6. **Visible fraction starts high, dips in the middle, rises at the
   end.** The U-shape is present in all models:
   - 3L: 24% → 19% → 58%
   - 6L: 48% → 39% → 32% → 33% → 34% → 61%
   - 7L: 51% → 32% → 30% → 30% → 29% → 33% → 74%

7. **Deeper models achieve higher P(d).** 39% → 45% → 70% → 82%.
   But the mechanism is the same — more depth gives more processing
   steps, not a different strategy.

### The tool confirms: the pipeline is universal

The functional summary makes the depth comparison possible in one
reading. Without the tool, this comparison required 4 separate logit
lens analyses, 4 attention traces, and manual synthesis. With the tool,
the patterns are visible immediately.
