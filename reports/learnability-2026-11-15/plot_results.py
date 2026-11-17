"""Plot saved, audited accuracy values; does not fit or select a model."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument("--results",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
data=json.loads(args.results.read_text())["eligible_dev"]
args.output.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
fig,axes=plt.subplots(1,2,figsize=(10.2,4.8),sharey=True)
xs=[0,1,2];sizes=[1800,6000,18000]
for ax,role,title in zip(axes,["qwen9b","startlux9b"],["Qwen3.5-9B fallback","StartLux-9B fallback (secondary)"]):
 for arm,label,color,marker in [("front","Front information only","#2463A6","o"),("text","Front + frozen text features","#B9482C","s")]:
  values=[100*data["cells"][f"{role}/n{n}/{arm}"]["accuracy"] for n in sizes]
  ax.plot(xs,values,label=label,color=color,marker=marker,lw=2,ms=6)
  for x,y in zip(xs,values):ax.annotate(f"{y:.2f}",(x,y),xytext=(0,-16 if role=="startlux9b" and arm=="front" else 7),textcoords="offset points",ha="center",fontsize=9,color=color)
 base=100*data["baselines"]["direct_startlux4b"]["accuracy"]
 ax.axhline(base,color="#606A73",ls="--",lw=1.2,label="Direct StartLux-4B (72.71%)")
 ax.set_title(title,loc="left",fontsize=12,pad=12)
 ax.set_xticks(xs,["1,800","6,000","18,000"])
 ax.set_xlabel("Fitting labels")
 ax.set_ylim(69.9,75.2);ax.set_xlim(-.18,2.18)
 ax.grid(axis="y",alpha=.16)
axes[0].set_ylabel("Accuracy (%)")
axes[1].axhline(100*data["baselines"]["direct_startlux9b"]["accuracy"],color="#34765A",ls=":",lw=1.5,label="Direct StartLux-9B (74.55%)")
handles,labels=axes[1].get_legend_handles_labels()
fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.11),ncol=2,frameon=False,fontsize=9)
fig.suptitle("ANLI eligible dev: routing at a fixed 10% offline quota",x=.08,ha="left",fontsize=14,fontweight="bold")
fig.text(.08,.035,"Primary contrast at 18k, Qwen fallback: text − front +0.92 pp; 95% CI [−0.11, +1.88] pp.\n1,737 inputs / 1,540 premise groups. Other plotted cells are descriptive; intervals condition on fitted models.",fontsize=9,color="#424A52")
fig.subplots_adjust(top=.80,bottom=.33,left=.08,right=.985,wspace=.12)
for ext in ["png","pdf"]:fig.savefig(args.output/f"learning_curve.{ext}",dpi=180,facecolor="white")
plt.close(fig)
