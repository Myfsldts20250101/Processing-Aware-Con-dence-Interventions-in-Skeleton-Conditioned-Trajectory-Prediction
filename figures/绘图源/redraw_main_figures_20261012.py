"""Redraw the four main figures from definitions and frozen summaries.
No training, inference, new sampling or statistical testing.
"""
from pathlib import Path
import os,csv,json,hashlib
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; PDF=ROOT/'pdf'
os.environ['MPLCONFIGDIR']=str(HERE/'matplotlib_cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
from matplotlib.ticker import PercentFormatter,MultipleLocator
BLUE='#596BD8'; PINK='#E83E8C'; GRAY='#D8DCE5'; INK='#273142'; GRID='#E0E3EA'; LIGHT='#F6F7FA'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#8C94A0','text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
meta={'date':'2026-10-12','source_style':'Pinar et al. IDA 2026, DOI 10.1177/1088467X261490986','scope':'Four main figures only; supplementary figures retained','no_training_or_inference':True,'no_new_sampling':True,'figures':{}}
def save(fig,name):
    fig.savefig(PDF/(name+'.pdf'),bbox_inches='tight',pad_inches=.08)
    fig.savefig(HERE/(name+'.svg'),bbox_inches='tight',pad_inches=.08)
    fig.savefig(PDF/(name+'.png'),dpi=400,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
def label(ax,x,y,s,size=9,ha='left',weight='normal',color=INK,va='center'):
    ax.text(x,y,s,fontsize=size,ha=ha,va=va,fontweight=weight,color=color,linespacing=1.35)
def box(ax,x,y,w,h,text,color=GRAY,fill=LIGHT,size=9,weight='normal'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.012',lw=1,edgecolor=color,facecolor=fill))
    label(ax,x+w/2,y+h/2,text,size,ha='center',weight=weight)
def arrow(ax,a,b,color=INK):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,lw=1,color=color,shrinkA=2,shrinkB=2))
# Figure 1: the processing comparison is the central object, not a new architecture.
fig,ax=plt.subplots(figsize=(7.4,6.2)); ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
label(ax,.015,.97,'a  Confidence assignment and processing',11,weight='bold')
label(ax,.015,.92,r'$J_t=(c_t>0.1)$;  $c_J=\mathrm{where}(J_t,c_d,c_t)$',10)
label(ax,.015,.875,'Target raw coordinates and trajectory history remain fixed.',9)
xs=[.025,.14,.29,.64]; heads=['Condition','Assigned conf.','Processing policy','Effective state']
for x,txt in zip(xs,heads): label(ax,x,.827,txt,8.5,weight='bold')
ax.plot([.015,.985],[.805,.805],color=GRID,lw=1)
rows=[('O',r'$c_t$','Official processing','Clean xy and masks',GRAY),('F',r'$c_J$','Replay clean structural state','Clean xy and masks',BLUE),(r'$N_J$',r'$c_J$','Recompute official processing','xy and masks may change',BLUE),('N',r'$c_d$ (full field)','Recompute official processing','xy and masks may change',PINK)]
for y,row in zip([.755,.685,.615,.545],rows):
    ax.add_patch(Rectangle((.015,y-.032),.97,.062,facecolor=LIGHT if row[0]!='O' else 'white',edgecolor='none'))
    ax.add_patch(Rectangle((.015,y-.032),.006,.062,facecolor=row[-1],edgecolor='none'))
    for x,txt in zip(xs,row[:4]): label(ax,x+.006,y,txt,8.5,weight='bold' if x==xs[0] else 'normal')
label(ax,.015,.468,'b  One frozen predictor, evaluated separately for each condition',10.5,weight='bold')
for x,w,txt in [(.025,.19,'Trajectory + skeleton\nencoders'),(.285,.18,'Gated\ncross-attention'),(.535,.18,'Condition prior\n'+r'$z_c=\mu_c$'),(.785,.19,'Fusion decoder\none trajectory')]:
    box(ax,x,.345,w,.085,txt,size=8.5)
for a,b in [((.215,.387),(.285,.387)),((.465,.387),(.535,.387)),((.715,.387),(.785,.387))]:arrow(ax,a,b)
label(ax,.5,.307,'Same weights; learned features and prior means may change.',8.5,ha='center')
label(ax,.015,.252,'c  Two conditional contrasts, two distinct measurements',10.5,weight='bold')
box(ax,.025,.139,.445,.07,r'$F\rightarrow N_J$: same assignment; change processing',BLUE,'#F0F2FC',8.5)
box(ax,.525,.139,.445,.07,r'$N_J\rightarrow N$: official rules; extend assignment',PINK,'#FDF0F6',8.5)
label(ax,.025,.088,r'Output: $R_{ij}=\mathrm{ADE}(\widehat Y_j,\widehat Y_i)$',9)
label(ax,.525,.088,r'Error: $\Delta_{ij}=L_j-L_i$,  $L_c=\mathrm{ADE}(\widehat Y_c,Y)$',9)
label(ax,.5,.035,'GT enters error measurement only. Masks need not agree under the same official rules.',8,ha='center')
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
save(fig,'correspondence_diagnostic_overview')
meta['figures']['overview']={'type':'Definition diagram','inputs':'Manuscript O/F/NJ/N definitions; no measured values added'}
# Figure 2: populations, repetitions and interface limits.
fig,ax=plt.subplots(figsize=(7.4,4.8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
label(ax,.02,.96,'a  Freeze the evaluation population',10.5,weight='bold')
box(ax,.03,.76,.405,.13,'Moving test stratum\n29,158 windows',GRAY,size=10,weight='bold')
arrow(ax,(.232,.755),(.232,.705))
box(ax,.03,.57,.405,.13,'Skeleton-available targets\n15,603 windows',GRAY,size=10)
label(ax,.232,.535,'BV1 passes all five frozen donor maps',8.3,ha='center')
arrow(ax,(.232,.51),(.232,.465))
box(ax,.03,.325,.405,.135,'Frozen '+r'$S_{\mathrm{BV}}$'+'\n15,160 mutually valid windows',BLUE,'#F0F2FC',10,weight='bold')
label(ax,.232,.278,'97.16% of available targets\n51.99% of the moving stratum',8.5,ha='center')
label(ax,.52,.96,'b  Evaluate shared keys and assignments',10.5,weight='bold')
box(ax,.53,.76,.44,.13,'8 training checkpoints × 5 donor maps\n40 crossed cells',BLUE,'#F0F2FC',10,weight='bold')
arrow(ax,(.75,.755),(.75,.705))
box(ax,.53,.57,.44,.13,'15,160 windows per cell\n606,400 repeated cell-window records',GRAY,size=9.3)
box(ax,.53,.325,.44,.135,'HST boundary: 4,372 candidates − 384\n= 3,988 windows in 17 scenes\nOne checkpoint × one donor',PINK,'#FDF0F6',9)
label(ax,.53,.275,'Different interface, subset and repetition structure',8.2)
ax.plot([.02,.98],[.215,.215],color=GRID,lw=1)
label(ax,.02,.16,'Initial content audit: 7,446 of 29,158 nominal replacements were unchanged (25.54%).',8.7)
label(ax,.02,.085,'The initial audit is a separate run. Repeated records are not independent windows or training runs.',8.1)
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
save(fig,'correspondence_protocol')
meta['figures']['protocol']={'type':'Population and repetition diagram','source':'Manuscript frozen counts; E0 audit explicitly separate from four-condition run'}
# Figure 3: same source means and min-max donor ranges, now separate panels.
p=HERE/'nj_policy_support_cell_means.csv'; rows=list(csv.DictReader(p.open(encoding='utf-8',newline='')))
seeds=list(range(42,50));donors=list(range(42,47)); assert len(rows)==40
fig,axs=plt.subplots(1,2,figsize=(7.4,3.65),sharex=True,sharey=True)
stats={}
for ax,key,col,title,mark in zip(axs,['delta_policy','delta_support'],[BLUE,PINK],['a  Processing: '+r'$N_J-F$','b  Assignment support: '+r'$N-N_J$'],['o','s']):
    vals=np.array([[float(next(r[key] for r in rows if int(r['train_seed'])==s and int(r['donor_map'])==d)) for d in donors] for s in seeds]); means=vals.mean(axis=1); yy=np.arange(8)
    for i,v in enumerate(vals): ax.scatter(v,np.full(5,i),s=10,color=col,alpha=.28,zorder=2)
    ax.errorbar(means,yy,xerr=np.vstack([means-vals.min(axis=1),vals.max(axis=1)-means]),fmt=mark,color=col,markersize=5,elinewidth=1.2,capsize=3,zorder=3)
    ax.axvline(0,color='#687181',lw=.85,ls='--'); ax.set_facecolor(LIGHT);ax.set_title(title,loc='left',pad=12)
    ax.set_yticks(yy,labels=seeds);ax.set_xlabel(r'Signed GT $\Delta$ADE');ax.grid(axis='x',color=GRID,lw=.7);ax.set_axisbelow(True);ax.set_ylim(7.65,-.65)
    ax.xaxis.set_major_locator(MultipleLocator(.01));ax.tick_params(axis='both',labelsize=8.5)
    ax.text(.02,1.015,f'{means.mean():.4f} ± {means.std(ddof=1):.4f} across seeds; {(means>0).sum()}/8 positive',transform=ax.transAxes,fontsize=8.2)
    stats[key]={'seed_means':means.tolist(),'mean':float(means.mean()),'sample_sd':float(means.std(ddof=1))}
axs[0].set_ylabel('Training seed');axs[0].set_xlim(-.009,.034)
fig.subplots_adjust(left=.09,right=.99,bottom=.19,top=.82,wspace=.17)
fig.text(.5,.035,'Large markers: five-donor means  |  Whiskers: donor min–max  |  Small markers: donor cell means',ha='center',fontsize=8)
save(fig,'correspondence_nj_policy_support')
meta['figures']['policy_support']={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'statistics':stats,'whiskers':'Donor min-max, not confidence intervals'}
# Figure 4: redraw archived bins without redrawing the historical sample.
p=HERE/'F_response_deciles_archived_sample_20261005.csv'; rows=list(csv.DictReader(p.open(encoding='utf-8',newline='')))
n=np.array([int(r['n_sampled_cell_window_records']) for r in rows]); assert (n==8000).all() and n.sum()==80000
vals=[np.array([float(r[k]) for r in rows]) for k in ['improved_fraction','in_band_fraction','worsened_fraction']]; assert np.allclose(sum(vals),1,atol=1e-14,rtol=0)
fig,ax=plt.subplots(figsize=(7.4,4.45));yy=np.arange(10);left=np.zeros(10)
for data,col,txt in zip(vals,[BLUE,GRAY,PINK],[r'Improved: $\Delta$ADE < −0.01',r'Band: $|\Delta$ADE| ≤ 0.01',r'Worsened: $\Delta$ADE > 0.01']):
    ax.barh(yy,data,left=left,height=.75,color=col,label=txt,edgecolor='white',lw=.4)
    for y,l,v in zip(yy,left,data):
        if v>=.085: ax.text(l+v/2,y,f'{v*100:.1f}%',ha='center',va='center',color='white' if col!=GRAY else INK,fontsize=8.3)
    left+=data
ax.set_yticks(yy,labels=range(1,11));ax.set_ylim(9.6,-.6);ax.set_xlim(0,1);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_xlabel('Fraction of sampled cell-window records');ax.set_ylabel('Prediction-response decile (low to high)');ax.set_title('Two-sided error changes across response deciles',loc='left',pad=43)
ax.legend(ncol=3,loc='lower left',bbox_to_anchor=(-.01,1.035),fontsize=7.6,frameon=False,handlelength=1.15,columnspacing=1.2)
ax.set_facecolor(LIGHT);ax.grid(axis='x',color=GRID,lw=.6);ax.set_axisbelow(True)
fig.subplots_adjust(left=.115,right=.985,bottom=.19,top=.78)
fig.text(.55,.035,'Archived pool: 80,000 repeated records; 8,000 per decile. The band is descriptive.',ha='center',fontsize=8)
save(fig,'correspondence_u1f_response_deciles')
meta['figures']['response_deciles']={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'n_records':80000,'records_per_bin':8000,'source':'Stored archived bin counts; original sample-row list unavailable','band':'Descriptive, not equivalence or no effect'}
(HERE/'style_redraw_metadata_20261012.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Four main figures redrawn; original numerical sources retained.')
