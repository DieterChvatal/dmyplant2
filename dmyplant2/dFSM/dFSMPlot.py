import pandas as pd
import numpy as np
import bokeh
from bokeh.models import Span, Text, Label
from dmyplant2 import dbokeh_chart, bokeh_show, add_dbokeh_vlines, add_dbokeh_hlines

def FSMPlot_Start(fsm,startversuch, data, vset, dset, figsize=(16,10)):
    von_dt=pd.to_datetime(startversuch['starttime']); von=int(von_dt.timestamp())
    bis_dt=pd.to_datetime(startversuch['endtime']); bis=int(bis_dt.timestamp())

    ftitle = f"{fsm._e} ----- Start {startversuch['no']} {startversuch['mode']} | {'SUCCESS' if startversuch['success'] else 'FAILED'} | {startversuch['starttime'].round('S')}"
    #print(ftitle, end=' ')
    print(f"von: {von_dt.strftime('%d.%m.%Y %H:%M:%S')} bis: {bis_dt.strftime('%d.%m.%Y %H:%M:%S')}")

    #vset = fsm._data_spec + ['Hyd_PressCrankCase','Hyd_PressOilDif','Hyd_PressOil','Hyd_TempOil']

    #data = fsm.get_cycle_data2(startversuch, max_length=None, min_length=None, cycletime=1, silent=False, p_data=vset)
    #fsm._debug(int(startversuch['starttime'].timestamp() * 1000),int(startversuch['endtime'].timestamp() * 1000),data,'data')


    #fsm.disp_result(startversuch)
    al_lines = fsm.disp_alarms(startversuch)
    w_lines = fsm.disp_warnings(startversuch)
    fig = dbokeh_chart(data, dset, title=ftitle, grid=False, figsize=figsize, style='line', line_width=0)

    add_dbokeh_vlines(al_lines,fig,line_color='purple', line_dash='dashed', line_alpha=1, line_width=2)
    add_dbokeh_vlines(w_lines,fig,line_color='brown', line_dash='dashed', line_alpha=1, line_width=2)
    add_dbokeh_vlines(fsm.states_lines(startversuch),fig,line_color='red', line_dash='solid', line_alpha=0.4)
                            
    fig.add_layout(Span(location=fsm._e['Power_PowerNominal'],dimension='width',x_range_name='default', y_range_name='0',line_color='red', line_dash='solid', line_alpha=0.4)) 
    if 'maxload' in startversuch:
        if startversuch['maxload'] == startversuch['maxload']:
            fig.add_layout(Span(location=startversuch['maxload'],dimension='width',x_range_name='default', y_range_name='0',line_color='red', line_dash='solid', line_alpha=0.4)) 
    fig.add_layout(Span(location=1500,dimension='width',x_range_name='default', y_range_name='1',line_color='blue', line_dash='solid', line_alpha=0.4)) 

    return fig
