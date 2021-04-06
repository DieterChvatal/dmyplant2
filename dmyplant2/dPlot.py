
# Standard Library imports
import arrow
from datetime import datetime
from itertools import cycle
import pandas as pd
import numpy as np
from pprint import pprint as pp
import statistics
import sys
import time
import traceback

# Third party imports
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as dates

#Bokeh imports
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure, output_file, show
from bokeh.models import LinearAxis, Range1d, HoverTool
from bokeh.core.validation import check_integrity
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource

# Load Application imports
from dmyplant2.dReliability import demonstrated_reliability_sr
import dmyplant2

def _idx(n, s, e, x):
    return int(n * (x - s) / (e - s)+1)


def demonstrated_Reliabillity_Plot(vl, beta=1.21, T=30000, s=1000, ft=pd.DataFrame, cl=[10, 50, 90], xmin=None, xmax=None, factor=2.0, ymax=24000):
    """Plot the demonstrated Reliability of the specified validation fleet

    Example:

    ....

    # load input data from files
    dval = pd.read_csv("input2.csv",sep=';', encoding='utf-8')
    dval['val start'] = pd.to_datetime(dval['val start'], format='%d.%m.%Y')
    failures = pd.read_csv("failures.csv",sep=';', encoding='utf-8')
    failures['date'] = pd.to_datetime(failures['date'], format='%d.%m.%Y')

    dmyplant2.demonstrated_Reliabillity_Plot(vl,
            beta=1.21, T=30000, s=1000, ft=failures, cl=[10,50,90], factor=1.3);

    ...


    Args:
        vl ([dmyplant2.Validation class]): [Class with several function around the validation fleet]
        beta (float, optional): [Weibull beta parameter]. Defaults to 1.21.
        T (int, optional): [Runtime for Assessment of Reliabiliy, calculated with LIPSON Method]. Defaults to 30000.
        s (int, optional): [number of points to plot]. Defaults to 1000.
        ft ([type], optional): [pd.DataFrame with observed failures]. Defaults to pd.DataFrame.
            required Columns: date;failures;serialNumber;comment
        cl (list, optional): [list with relialibilty lines for specific confidence levels to plot,
            Numbers between 0 and 100]. Defaults to [10, 50, 90].
        xmin ([timestamp], optional): [left margin of x-axis]. Defaults to None.
        xmax ([timestamp], optional): [right margin of x-axis]. Defaults to None.
        factor (float, optional): [Extrapolation factor]. Defaults to 2.0.
        ymax (int, optional): [right y-axis max value]. Defaults to 24000.

    Raises:
        ValueError: [Time Range not properly specified]
    """
    # define milestones
    start_ts = vl.valstart_ts if xmin == None else xmin  # val start

    # the end of the Plotting interval
    if xmax:
        last_ts = xmax
    else:
        if factor:
            factor = max(factor, 1.0)  # no factor < 1.0 allowed
            elapsed = vl.now_ts - start_ts
            last_ts = start_ts + factor * elapsed
        else:
            raise ValueError("Error in timerange specification.")

    fcol = 'grey'

    # calculate the x axis timerange first
    tr = demonstrated_reliability_sr(vl,
                                     start_ts, last_ts, beta=beta, size=s, ft=ft)[0]  # timestamp x axis start .. end

    # determine the array - index of 'now'
    n_i = _idx(s, start_ts, last_ts, vl.now_ts)

    # create Timerow from Start to 'now'
    n_tr = tr[0:n_i:1]

    # convert to datetime dates - start .. last
    dtr = [datetime.fromtimestamp(t) for t in tr]
    # calculate demonstrated reliability curves for the complete period,
    # confidence intervals CL :
    rel = {c: demonstrated_reliability_sr(vl, start_ts, last_ts,
                                          CL=c/100.0, beta=beta, size=s, ft=ft, T=T)[1] for c in cl}

    # convert to datetime dates - start .. now
    n_dtr = [datetime.fromtimestamp(t) for t in n_tr]
    # copy demontrated reliability values for the validation period up to now:
    n_rel = {c: rel[c][0:n_i:1] for c in cl}

    # define the PLOT
    fig, ax1 = plt.subplots(  # pylint: disable=unused-variable
        figsize=(12, 8), constrained_layout=True)
    # fig, (ax1, ax3) = plt.subplots(2, figsize=(6, 6))

    color = 'tab:red'
    ax1.set_xlabel('date')
    ax1.set_ylabel('Demonstrated Reliability [%]', color=color)
    ax1.set_title('Demonstrated Reliability [%]')

    # now plot the demonstrated reliability curves:
    for CL in cl:
        # complete interval in color fcal
        ax1.plot(dtr, rel[CL], color=fcol, linestyle='-', linewidth=0.5)
        # the current validation interval in multiple colors
        ax1.plot(n_dtr, n_rel[CL], color='red', linestyle='-', linewidth=0.7)

    # define the axis ticks
    ax1.tick_params(axis='y', labelcolor=color)

    # and the axis scales
    ax1.axis((datetime.fromtimestamp(start_ts),
              datetime.fromtimestamp(last_ts), 0, 100))

    # define axis intervals y ...
    ax1.yaxis.set_major_locator(ticker.LinearLocator(13))

    # and x - axis
    locator = dates.AutoDateLocator()
    locator.intervald[dates.MONTHLY] = [1]
    ax1.xaxis.set_major_locator(locator)

    # show a grid
    ax1.grid(color='lightgrey')

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    ax2.axis((datetime.fromtimestamp(start_ts),
              datetime.fromtimestamp(last_ts), 0, ymax))
    color = 'tab:blue'
    # the x-label was handled with ax1
    ax2.set_ylabel('hours [h]', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.yaxis.set_major_locator(ticker.LinearLocator(13))

    # and plot the linearized engine runtime lines vs the 2nd axis
    for e in vl.engines[:]:
        # print(e.Name, e._d['Engine ID'], e._d['val start'], e._d['oph parts'])
        # complete interval in color fcal
        y = [e.oph(t) for t in tr]
        ax2.plot(dtr, y, linewidth=0.5, color=fcol)
        # the current validation interval in multiple colors
        n_y = [e.oph(t) for t in n_tr]
        ax2.plot(n_dtr, n_y, label=f"{e.Name} {e._d['Engine ID']}")

    # NOW plot some Orientation Lines and Test into the Plot

    # possible runtime line
    y = [(t-start_ts) / 3600.0 for t in tr]
    ax2.plot(dtr, y, color='grey', linestyle='--', linewidth=0.7)

    # today line
    ax1.axvline(datetime.now(), color='red',
                linestyle='--', linewidth=0.7)

    # Point of demonstrated reliability at
    # highest Confidence Level, today
    myrel_y = float(
        rel[max(cl)][int((vl.now_ts-start_ts)/(last_ts - start_ts)*s-1)])
    myrel_x = datetime.fromtimestamp(vl.now_ts)
    ax1.scatter(myrel_x, myrel_y, marker='o', color='black', label='point')
    txt = f"CL {max(cl)}%@{T}\nbeta={beta}\nR={myrel_y:.1f}%"

    # some statistical Information.
    myrel_txt_x = datetime.fromtimestamp(vl.now_ts + 200000)
    ax1.text(myrel_txt_x, myrel_y - 9, txt)
    ax1.axis((datetime.fromtimestamp(start_ts),
              datetime.fromtimestamp(last_ts), 0, 120))
    # oph Fleet Leader
    fl = [e.oph2(vl.now_ts) for e in vl.engines]
    fl_point_x = datetime.fromtimestamp(vl.now_ts)
    ax2.scatter(fl_point_x, max(fl), marker='o', color='black', label='point')
    fl_txt_x = datetime.fromtimestamp(vl.now_ts + 200000)
    txt = f'{len(fl)} engines\nmax {max(fl):.0f}h\ncum {sum(fl):.0f}h\navg {statistics.mean(fl):.0f}h\n{arrow.now("Europe/Vienna").format("DD.MM.YYYY HH:mm")}'
    ax2.text(fl_txt_x, max(fl) - T/7, txt)

    # def on_plot_hover(event):
    #     # Iterating over each data member plotted
    #     for curve in ax2.get_lines():
    #         # Searching which data member corresponds to current mouse position
    #         if curve.contains(event)[0]:
    #             print("over %s" % curve.get_gid())

    # plt.legend()
    # fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)

    # TATAAAAH!
    plt.show()


def chart(d, ys, x='datetime', title=None, grid=True, legend=True, *args, **kwargs):
    """Generate Diane like chart with multiple axes

    example:
    .....

    dat = {
        161: ['CountOph','h'],
        102: ['PowerAct','kW'],
        107: ['Various_Values_SpeedAct','U/min'],
        217: ['Hyd_PressCrankCase','mbar'],
        16546: ['Hyd_PressOilDif','bar']
    }

    df = mp.hist_data(
        e.id,
        itemIds=dat,
        p_from=arrow.get('2021-03-05 05:28').to('Europe/Vienna'),
        p_to=arrow.get('2021-03-05 05:30').to('Europe/Vienna'),
        timeCycle=1)


    dmyplant2.chart(df, [
    {'col': ['PowerAct'],'ylim': [0, 5000]},
    {'col': ['Various_Values_SpeedAct'],'ylim': [0, 2500], 'color':'darkblue'},
    {'col': ['CountOph'],'ylim': [0, 500]},
    {'col': ['Hyd_PressCrankCase'],'ylim': [-40, 60]},
    {'col': ['Hyd_PressOilDif'],'ylim': [0, 1]}
    ],
    title = e,
    grid = False,
    figsize = (14,10))

    .....

    Args:
        d (pd.dataFrame): Data , e.g downloaded by engine.batch_hist_dataItems(...)
        ys ([list of dicts]): the DataFrame d columns to plot
        x (str, optional): x-axis column as string. Defaults to 'datetime'.
        title (str, optional): Main Title of figure. Defaults to None.
        grid (bool, optional): displaygrid on left axis. Defaults to True.
        legend (bool, optional): legend. Defaults to True.
    """
    # for entry in kwargs.items():
    #     print("Key: {}, value: {}".format(entry[0], entry[1]))

    fig, ax = plt.subplots(*args, **kwargs)

    axes = [ax]
    ax.tick_params(axis='x', labelrotation=30)

    if grid:
        ax.grid()
    if title:
        ax.set_title(title)

    for y in ys[1:]:
        # Twin the x-axis twice to make independent y-axes.
        axes.append(ax.twinx())

    fig.subplots_adjust(top=0.9)
    fig.subplots_adjust(left=0.1)

    extra_ys = len(axes[2:])

    # Make some space on the right side for the extra y-axes.
    if extra_ys > 0:
        if extra_ys > 6:
            print('you are being ridiculous')
            raise ValueError('too many Extra Axes')
        else:
            temp = 0.9 - extra_ys * 0.05

        # print('you are being ridiculous')
        fig.subplots_adjust(right=temp)
        right_additive = 0.065 / temp

        # Move the last y-axis spine over to the right by x% of the width of the axes
        for i, ax in enumerate(axes[2:]):
            ax.spines['right'].set_position(
                ('axes', 1.0 + right_additive * (i+1)))
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # To make the border of the right-most axis visible, we need to turn the frame
        # on. This hides the other plots, however, so we need to turn its fill off.

    cols = []
    lines = []
    # line_styles = cycle(['-', '-', '-', '--', '-.', ':', 'dotted', ',', 'o', 'v', '^', '<', '>',
    #                     '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'])
    line_styles = cycle(['-', '-', '-', '--', '-.', ':'])

    colors = cycle(matplotlib.rcParams['axes.prop_cycle'])
    for ax, y in zip(axes, ys):
        ls = next(cycle(line_styles))
        if len(y['col']) == 1:
            col = y['col'][0]
            cols.append(col)
            if 'color' in y:
                color = y['color']
            else:
                color = next(cycle(colors))['color']
            lines.append(ax.plot(d[x], d[col],
                                 linestyle=ls, label=col, color=color))
            ax.set_ylabel(col, color=color)
            if 'ylim' in y:
                ax.set_ylim(y['ylim'])
            ax.tick_params(axis='y', colors=color)
            ax.spines['right'].set_color(color)
        else:
            for col in y['col']:
                if 'color' in y:
                    color = y['color']
                else:
                    color = next(cycle(colors))['color']
                lines.append(
                    ax.plot(d[x], d[col], linestyle=ls, label=col, color=color))
                cols.append(col)
            llabel = ', '.join(y['col'])
            if len(llabel) > 90:
                llabel = llabel[:87] + ' ..'
            ax.set_ylabel(llabel)
            if 'ylim' in y:
                ax.set_ylim(y['ylim'])
            ax.tick_params(axis='y')
    axes[0].set_xlabel(d.index.name)
    lns = lines[0]
    for l in lines[1:]:
        lns += l
    labs = [l.get_label() for l in lns]
    if legend:
        axes[0].legend(lns, labs, loc=0)

def bokeh_chart(source, pltcfg, x_ax='datetime', title=None, grid=True, legend=True, style='line', x_range=None, y_range=None, *args, **kwargs):
    """Generate interactive Diane like chart with multiple axes

    Args:
        source (bokeh.ColumnDataSource): Data , e.g downloaded by engine.batch_hist_dataItems(...)
        pltcfg ([list of dicts]): the source columns to plot, and range of y-axis
        x_ax (str, optional): x-axis column as string. Defaults to 'datetime'.
        title (str, optional): Main Title of figure. Defaults to None.
        grid (bool, optional): display grid. Defaults to True.
        legend (bool, optional): legend. Defaults to True.  
        style (str, optional): style of markers, options i.e. 'line', 'circle'
            circle necessary to enable linked brushing (selection of datapoints)
        x_range (bokeh.figure.x_range, optional): x_range of different bokeh-plot; used to connect x-axis limits
        y_range (bokeh.figure.y_range, optional): y_range of different bokeh-plot; used to connect y-axis limits


    Returns:
        bokeh.plotting.figure: Bokeh plot ready to plot or embed in a layout

    example:
    .....
    from bokeh.io import push_notebook, show, output_notebook
    from bokeh.plotting import figure, output_file, show
    from bokeh.models import LinearAxis, Range1d, HoverTool
    from bokeh.layouts import column, row, gridplot
    from bokeh.models import ColumnDataSource
    from itertools import cycle
    import dmyplant2
    import arrow

    import pandas as pd
    import numpy as np
    import traceback
    import matplotlib
    import sys
    import warnings
    

    dmyplant2.cred()
    mp = dmyplant2.MyPlant(0)

    # Version mittels Validation Instance 
    dval = dmyplant2.Validation.load_def_csv('input.csv')
    vl = dmyplant2.Validation(mp, dval, cui_log=True)
    e = vl.eng_serialNumber(1145166)

    print(f"{e} {e.id}")
    dat = {
        161: ['CountOph','h'],
        102: ['PowerAct','kW'],
        107: ['Various_Values_SpeedAct','U/min'],
        217: ['Hyd_PressCrankCase','mbar'],
        16546: ['Hyd_PressOilDif','bar']
    }

    df = mp.hist_data(
        e.id,
        itemIds=dat,
        p_from=arrow.get('2021-03-05 05:28').to('Europe/Vienna'),
        p_to=arrow.get('2021-03-05 05:30').to('Europe/Vienna'),
        timeCycle=1)


    pltcfg = [
        {'col': ['PowerAct', 'Various_Values_SpeedAct'], 'unit':'1/min'},  # , 'ylim':(0, 5000) 
        {'col': ['Various_Values_SpeedAct'], 'ylim':(0, 1800), 'unit':'1/min'},
        {'col': ['Hyd_PressCrankCase'], 'ylim': [-50, 20], 'unit':'mbar'},
        {'col': ['Hyd_PressOilDif'], 'ylim':(0, 1), 'unit':'bar'}
    ]

    output_notebook()

    df.loc['2021-03-05 05:00':'2021-03-05 06:00']

    title=e._info.get('Validation Engine')

    source = ColumnDataSource(df)
    output_file(title+'.html')
    p=bokeh_chart(source, pltcfg, title=title)

    show(p)
    """

    TOOLS = 'pan, box_zoom, wheel_zoom, box_select, reset, save' #select Tools to display
    colors = cycle(matplotlib.rcParams['axes.prop_cycle']) #colors to use for plot
    linewidth = 2

    if (x_ax == 'datetime'): #seperate constructors for object for datetime or no datetime x-axis
        p = figure(
        plot_width=800,
        plot_height=600,
        x_axis_label='datetime',
        x_axis_type='datetime',
        x_range=x_range,
        y_range=y_range,
        tools=TOOLS
        )
    else:
        p = figure(
            plot_width=800,
            plot_height=600,
            x_axis_label=x_ax,
            tools=TOOLS,
            x_range=x_range,
            y_range=y_range
        )

    if grid==True:
        p.grid.grid_line_color = None

    p.yaxis.visible = False
    tooltips = []
    for i, y in enumerate(pltcfg):

        color = next(cycle(colors))['color']
        if y.get('ylim'):
            ylim = list(y['ylim'])
            p.extra_y_ranges[str(i)] = Range1d(start=ylim[0], end=ylim[1])
        else: #if no ylim defined, calculate min, max and set borders
            max_val=0 
            min_val=0
            for entry in y['col']:
              maxi=np.amax(source.data[entry])
              mini=np.amin(source.data[entry])
              if maxi>max_val:
                 max_val=maxi
              if mini<min_val:
                 min_val=mini
            p.extra_y_ranges[str(i)] = Range1d(min_val*1.15, max_val*1.15)
            #p.extra_y_ranges[str(i)] = Range1d(
                #min(0, 1.15 * df[y['col']].min().min()), 1.15 * df[y['col']].max().max()) Implementation with DataFrame
        for col in y['col']:
            if 'color' in y:
                color = y['color']
            else:
                color = next(cycle(colors))['color']
            func = getattr(p, style) #to choose between different plotting styles
            func(source=source, x=x_ax, y=col, #circle or line
            color=color, y_range_name=str(i), legend_label=col, line_width=linewidth)
            tooltips.append((col, '@'+col + '{0.2 f} '+y['unit']))  # or 0.0 a

        
        
        llabel = ', '.join(y['col'])+' ['+y['unit']+']'
        
        if len(llabel) > 90:
                llabel = llabel[:86] + ' ...'
        if len(y['col']) > 1:
            color = 'black'
        p.add_layout(LinearAxis(y_range_name=str(i),
                            axis_label=llabel, axis_label_text_color=color), 'left')

    p.add_tools(HoverTool(tooltips=tooltips,
                           mode='mouse'))  # mode=vline -> display a tooltip whenever the cursor is vertically in line                                              with a glyph
    p.toolbar.active_scroll = p.select_one('WheelZoomTool')

    p.legend.click_policy='hide' #hides graph when you click on legend, other option mute (makes them less visible)
    p.legend.location = 'top_left'

    p.title.text = title
    p.title.text_font_size = '20px' 

    return p

if __name__ == '__main__':
    pass
