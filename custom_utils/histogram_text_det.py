
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


labels = ['SROIE', 'SROIE']
# labels = ['IC15'] * 3
y1 = [83.25, 83.09]
y2 = [84.87, 84.47]
# plt.rcParams['font.family'] = ['Times New Roman']

fig,ax = plt.subplots(1,1,figsize=(4,3.5))

x = np.arange(0, len(labels))

width = 0.2 # 柱子宽度

label_font = {
    'weight':'bold',
    'size':14,
    'family':'simsun'
}

ax.tick_params(which='major',direction='in',length=5,width=1,labelsize=11,bottom=False)
# labelrotation=0 标签倾斜角度
ax.tick_params(axis='x',labelsize=11,bottom=False,labelrotation=0)


ax.set_xticks(x)
x_max = 90
start_value = 80
ax.set_ylim(ymin = start_value ,ymax = x_max)
# 0 - 1800 ，200为一个间距
ax.set_yticks(np.arange(start_value, x_max +1 , 5))
ax.set_ylabel('F1(%)')
# ax.set_ylabel('F1(%)',fontdict=label_font)

ax.set_xticklabels(labels)
# ax.set_xticklabels(labels,fontdict=label_font)

plt.grid(axis='y', alpha=0.75)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
ax.tick_params(bottom=False,top=False,left=False,right=False)  #移除全部刻度线
ax.spines['bottom'].set_color('gray') #为底部轴添加颜色
# ax.spines['bottom'].set_color('#F5F5F5') #为底部轴添加颜色
# 上下左右边框线宽
linewidth = 1 # 2
for spine in ['top','bottom','left','right']:
    ax.spines[spine].set_linewidth(linewidth)

#ax.legend(markerscale=10,fontsize=12,prop=legend_font)
ax.legend(markerscale=10,fontsize=12)


'''
# 设置有边框和头部边框颜色为空right、top、bottom、left
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
'''


    # Add some text for labels, title and custom x-axis tick labels, etc.
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# colors = ['#9999FF','#58C9B9','#CC33CC','#D1B6E1','#99FF99','#FF6666']
# colors = ['#9999FF','#58C9B9','#CC33CC']
colors = ['#FCE6C9','#EED5D2','#B0E0E6']
# colors = ['#99FF99'] * 6
colors_o = ['#94AAD8'] * 6

color_tt = [(0.3098, 0.5059, 0.74120), (0.6078, 0.7333, 0.3490), (0.7490, 0.3137, 0.3020),
         (0.50, 0.50, 0.50), (0.93, 0.69, 0.13),  (0.30, 0.75, 0.93),
         (0.50, 0.39, 0.64), (0.15, 0.15, 0.15), (0.18, 0.64, 0.54)]


label_ohter = ['TRIE', 'Baseline']
# label_ohter = ['PAN', 'FCENet', 'DBNet']
# label_ohter = 'DBNet'
# rects1 = ax.bar(x - width/2, y1, width, label=label_ohter,ec='k',color=colors,lw=.8, hatch='/')
for i in range(len(labels)):
    rects1 = ax.bar(i - width/2, y1[i], width, label=label_ohter[i],ec='k',color=colors[i],lw=.8, hatch='/')
    autolabel(rects1)
# rects2 = ax.bar(x + width/2 + .05, y2, width, label='Ours',ec='k',color='white',lw=.8,hatch='\\')
rects4 = ax.bar(x + width/2 + .05, y2, width, label='Ours',ec='k',color=colors_o,lw=.8,hatch='\\')

autolabel(rects4)

# plt.legend(title=('1','2'))
# plt.legend()
plt.legend(loc='best',frameon=False) #去掉图例边框
# legend1 = ax.legend([rects1], ["line1"], loc="upper right", edgecolor = colors[2])
# ax.add_artist(legend1)
# ax.legend([rects1], ["DBNet"], edgecolor = colors[1])

# plt.legend([line_up, line_down], ['Line Up', 'Line Down'])
#best, 'upper left' , 'upper center', 'upper right',
# 'center left', 'center', 'center right',
# 'lower left', 'lower center', 'lower right']



fig.tight_layout()

pdf = PdfPages('figure_det.pdf')
pdf.savefig()
pdf.close()
# plt.savefig(r'./p1.png',dpi=500)
plt.show()
