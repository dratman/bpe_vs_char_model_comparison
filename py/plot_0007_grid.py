import json, collections
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = json.load(open('doc/0007_grid_results.json'))
def get(arm, it=20000, T=0.8):
    return next(r for r in R if r['arm']==arm and r['iter']==it and r['temp']==T)
arms = sorted({r['arm'] for r in R}, key=lambda a:(int(a.split('_')[0][1:]), int(a.split('_L')[1])))
out=[]
out.append('| arm | params | best val bpc | real% T0.8 | real% T0.5 | 1-4 | 5-7 | 8-10 | 11-14 | novel-real share (T0.8) | mean gen len (train) |')
out.append('|---|---|---|---|---|---|---|---|---|---|---|')
for a in arms:
    r=get(a); r5=get(a,T=0.5)
    b=' | '.join(f"{100*v['real']/v['n']:.0f}% (n={v['n']})" if v['n'] else f"- (n=0)" for v in r['buckets'].values())
    nov = r['real_val']/r['real'] if r['real'] else float('nan')
    out.append(f"| {a} | {r['n_params']:,} | {r['best_val_bpc']:.3f} | {100*r['real']/r['n']:.1f}% | {100*r5['real']/r5['n']:.1f}% | {b} | {100*nov:.1f}% | {r['mean_gen_len']:.2f} ({r['mean_train_len']:.2f}) |")
open('doc/0007_grid_table.md','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
col={1:'C0',2:'C1',4:'C2'}
for fn,key,yl in [('real','real','real-word % @20K (T0.8)'),('bpc','bpc','best val bpc')]:
    plt.figure(figsize=(6,4))
    for L in (1,2,4):
        xs=[get(a)['n_params'] for a in arms if a.endswith(f'_L{L}')]
        ys=[(100*get(a)['real']/1000 if key=='real' else get(a)['best_val_bpc']) for a in arms if a.endswith(f'_L{L}')]
        o=sorted(zip(xs,ys)); plt.plot(*zip(*o),'o-',color=col[L],label=f'{L} layer(s)')
    plt.xscale('log'); plt.xlabel('parameters'); plt.ylabel(yl); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(f'doc/figures/0007_{key}_vs_params.png',dpi=130); plt.close()
plt.figure(figsize=(8,4.5)); bks=['1-4','5-7','8-10','11-14']
for i,a in enumerate(arms):
    ys=[(100*get(a)['buckets'][b]['real']/get(a)['buckets'][b]['n'] if get(a)['buckets'][b]['n'] else float('nan')) for b in bks]
    plt.plot(bks,ys,'o-',label=a,color=plt.cm.tab20(i))
plt.xlabel('generated length (letters)'); plt.ylabel('real % @20K, T0.8'); plt.legend(fontsize=7,ncol=2); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig('doc/figures/0007_real_by_length.png',dpi=130)
with open('doc/0007_samples.md','w') as f:
    for a in arms: f.write(f"**{a}**: "+', '.join(get(a)['samples'])+'\n\n')
