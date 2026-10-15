# 0009 Mac Studio (MPS) memory/throughput probe

torch 2.5.1, macOS 26.5.2, MPS available True. Real py/model.py GPT, block 1024, vocab 8192, bias on, tied weights, dropout 0, AdamW unfused (wd 0.1, lr 6e-4, betas 0.9/0.95), random token ids, 2 warm-up + 10 timed steps, one fresh process per config. TFLOPS = 6 x params x tokens/s.

Memory cells: **driver_allocated (pool high-water) / current_allocated (post-step)** GiB. MPS has no peak-stats API; current understates (post-step), driver overstates (greedy pool). Then tokens/s / TFLOPS.

| size | mode | mb=4 | mb=8 | mb=16 | mb=32 |
|---|---|---|---|---|---|
| S125 | fp32 | 8.3/1.7 / 16.3k / 9.0 | 15.4/2.3 / 18.4k / 10.2 | 35.6/1.6 / 19.0k / 10.5 | 75.6/1.0 / 19.0k / 10.5 |
| S125 | bf16 | 8.3/2.1 / 16.3k / 9.0 | 15.4/1.7 / 17.9k / 9.9 | 35.6/1.4 / 19.0k / 10.5 | 75.6/1.0 / 19.0k / 10.5 |
| S125 | bf16+fp32attn | 9.3/2.2 / 16.2k / 9.0 | 15.4/1.7 / 18.4k / 10.2 | 35.6/1.4 / 18.9k / 10.5 | 75.6/1.0 / 18.9k / 10.4 |
| S125 | bf16native | 7.2/1.4 / 12.0k / 6.6 | 13.4/1.8 / 13.0k / 7.2 | 34.6/0.9 / 13.3k / 7.4 | 54.6/1.7 / 13.4k / 7.4 |
| M350 | fp32 | 23.4/5.4 / 5.9k / 11.0 | 36.6/6.0 / 6.3k / 11.8 | 59.1/4.6 / 6.4k / 12.0 | 191.6/4.6 / 1.0k / 1.9 |
| M350 | bf16 | 22.4/5.2 / 5.9k / 11.0 | 37.6/6.9 / 6.3k / 11.8 | 59.1/4.5 / 6.4k / 12.0 | 191.6/4.7 / 1.8k / 3.3 |
| M350 | bf16+fp32attn | 21.4/4.6 / 5.9k / 11.1 | 33.6/5.9 / 6.3k / 11.8 | 59.1/3.5 / 6.4k / 12.0 | 191.6/5.7 / 1.6k / 3.0 |
| M350 | bf16native | 20.3/4.0 / 4.0k / 7.5 | 42.1/4.3 / 4.2k / 7.9 | 91.1/3.4 / 4.3k / 8.1 | 180.6/3.7 / 4.3k / 8.1 |
| L760 | fp32 | 38.5/12.7 / 2.8k / 12.3 | 93.2/10.7 / 3.0k / 12.9 | - | - |
| L760 | bf16 | 36.5/10.7 / 2.8k / 12.3 | 93.2/10.7 / 3.0k / 12.9 | - | - |
| L760 | bf16+fp32attn | 36.5/10.2 / 2.8k / 12.3 | 93.2/10.8 / 3.0k / 12.8 | - | - |
| L760 | bf16native | 37.4/9.2 / 1.9k / 8.4 | 83.9/9.3 / 2.0k / 8.6 | - | - |
