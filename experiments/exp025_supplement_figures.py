"""Generate E025-aligned supplementary schematics without changing historical assets."""
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

P = {'navy':'#1f3b73','blue':'#2f7de1','blue_soft':'#eaf3ff','teal':'#1f9e8a','teal_soft':'#e8fbf7','green':'#34a853','green_soft':'#ecf9ef','orange':'#ef8a17','orange_soft':'#fff3e5','purple':'#7b61c9','purple_soft':'#f2ecff','red':'#d9534f','red_soft':'#fdeeee','gray':'#64748b','ink':'#18212f'}

def box(ax, x, y, w, h, face, edge, label, body='', fontsize=13):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.3,rounding_size=2', facecolor=face, edgecolor=edge, linewidth=1.5))
    ax.text(x+w/2, y+h-3.2, label, ha='center', va='top', fontsize=fontsize, fontweight='bold', color=edge)
    if body: ax.text(x+w/2, y+h/2-1.1, body, ha='center', va='center', fontsize=10.5, linespacing=1.35, color=P['ink'])

def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=16,linewidth=1.7,color=P['navy']))

def setup(title, subtitle):
    fig,ax=plt.subplots(figsize=(12,6.75)); fig.patch.set_facecolor('white')
    ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis('off')
    ax.text(50,96,title,ha='center',va='center',fontsize=25,fontweight='bold',color=P['navy'])
    ax.text(50,91,subtitle,ha='center',va='center',fontsize=11,color=P['gray'])
    return fig,ax

def save(fig, out, name):
    fig.savefig(out/(name+'.svg'),bbox_inches='tight',facecolor='white')
    fig.savefig(out/(name+'.png'),dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,required=True); out=p.parse_args().output_dir; out.mkdir(parents=True,exist_ok=True)
    fig,ax=setup('Evidence ladder and claim boundary','Historical mechanism diagnostics inform the model; E025 tests a shared simulator output')
    box(ax,3,61,28,21,P['blue_soft'],P['blue'],'1  Mechanism traces','E017–E018: dynamics\nand internal signatures')
    box(ax,36,61,28,21,P['teal_soft'],P['teal'],'2  Boundaries','E019–E022R: parameter, motif\nand replay-specific diagnostics')
    box(ax,69,61,28,21,P['purple_soft'],P['purple'],'3  Common probe','E024: protocol development\nE025: tuned, locked holdout')
    arrow(ax,(31.5,71.5),(35,71.5)); arrow(ax,(64.5,71.5),(68,71.5))
    box(ax,4,24,45,27,P['green_soft'],P['green'],'E025 result','Hebbian and resource-only variants exceed\nthe full model; shuffled replay is practically\nequivalent on the prespecified readout.')
    box(ax,53,24,43,27,P['red_soft'],P['red'],'Claim boundary','No demonstrated unique slow-writing advantage.\nNo direct biological validation of $M_b$ or\nunique molecular memory code.')
    ax.text(50,10,'S3 open-data bridges are exploratory; they do not measure $M_b$ or validate E025.',ha='center',va='center',fontsize=11,color=P['gray'])
    save(fig,out,'Fig_e025_s5_evidence_ladder')
    fig,ax=setup('Comparator evaluation in E025','A common output test complements mechanism-specific diagnostics')
    box(ax,3,30,27,51,P['blue_soft'],P['blue'],'Shared comparison','Ten specified families\n32 candidates per family\nShared probe and readout\nDevelopment / validation: 40 / 40\nHoldout: 100 topology draws')
    box(ax,36,48,28,33,P['purple_soft'],P['purple'],'E025 output test','Shared vs private probe change D\nPaired differences, ±0.02 margin\nNine families qualified;\neligibility-only unavailable')
    box(ax,70,48,27,33,P['green_soft'],P['green'],'Immediate outcome','Full model exceeds five families\nShuffled replay: practical\nequivalence; Hebbian and\nresource-only exceed full model')
    arrow(ax,(30.5,65),(35,65)); arrow(ax,(64.5,65),(69,65))
    box(ax,36,13,61,25,P['orange_soft'],P['orange'],'Historical mechanism diagnostics','E018 / E022 use mechanism-specific measures and proxies.\nThey describe internal signatures; they do not establish a unique\nadvantage on the E025 common readout.')
    ax.text(50,5,'Results apply to the tested implementations, adapters, search ranges and generated topologies.',ha='center',fontsize=10,color=P['gray'])
    save(fig,out,'Fig_e025_s5_comparator_comparison')

if __name__=='__main__': main()
