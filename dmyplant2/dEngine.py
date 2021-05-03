from datetime import datetime, timedelta
import math
from pprint import pprint as pp
import pandas as pd
import numpy as np
from dmyplant2.dMyplant import epoch_ts, mp_ts
from dmyplant2.dPlot import datastr_to_dict
import sys
import os
import pickle
import logging
import json
import arrow
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class MyPlantException(Exception):
    pass


class Engine:
    """
    Class to encapsulate Engine properties & methods
    """
    _sn = 0
    _picklefile = ''
    _properties = {}
    _dataItems = {}
    _k = None
    _P = 0.0
    _d = {}
    _info = {}

    def __init__(self, mp, eng):
        """Engine Constructor

        Args:
            mp (dmyplant2.Myplant): Myplant class instance
            eng (dict): Validation engine input data

        Doctest:
        >>> e = dmyplant2.Engine(mp, eng)
        >>> e.serialNumber
        '1320072'
        """
        # take engine Myplant Serial Number from Validation Definition
        self._mp = mp
        self._eng = eng
        self._sn = str(eng['serialNumber'])
        fname = os.getcwd() + '/data/' + self._sn
        self._picklefile = fname + '.pkl'    # load persitant data
        # self._lastcontact = fname + '_lastcontact.pkl'
        self._infofile = fname + '.json'

        # load info json & lastfetchdate
        try:
            with open(self._infofile) as f:
                self._info = json.load(f)
                if 'last_fetch_date' in self._info:
                    self._last_fetch_date = self._info['last_fetch_date']
                self._info = {**self._info, **self._eng}
                self._info['val start'] = arrow.get(
                    self._eng['val start']).timestamp()
        except:
            self._info = {**self._info, **self._eng}
        try:
            # fetch data from Myplant only on conditions below
            if self._cache_expired()['bool'] or (not os.path.exists(self._picklefile)):
                try:
                    local_asset = self._mp._asset_data(self._sn)
                    logging.debug(
                        f"{eng['Validation Engine']}, Engine Data fetched from Myplant")
                    self.asset = self._restructure(local_asset)

                    # add patch.json values
                    fpatch = os.getcwd() + '/patch.json'
                    if os.path.exists(fpatch):
                        with open(os.getcwd() + "/patch.json", "r", encoding='utf-8-sig') as file:
                            patch = json.load(file)
                            if self._sn in patch:
                                for k,v in patch[self._sn].items():
                                    if k in self.asset:
                                        self.asset[k] = {**self.asset[k], **v}
                                    else:
                                        self.asset[k] = v

                    self._last_fetch_date = epoch_ts(datetime.now().timestamp())
                except: 
                    raise
            else:
                with open(self._picklefile, 'rb') as handle:
                    self.__dict__ = pickle.load(handle)
        except FileNotFoundError:
            logging.debug(
                f"{self._picklefile} not found, fetch Data from MyPlant Server")
        else:
            logging.debug(
                f"{__name__}: in cache mode, load data from {self._sn}.pkl")
        finally:
            logging.debug(
                f"Initialize Engine Object, SerialNumber: {self._sn}")
            self._d = self._engine_data(eng)
            self._set_oph_parameter()
            self._save()

    def __str__(self):
        return f"{self._sn} {self._d['Engine ID']} {self.Name[:20] + (self.Name[20:] and ' ..'):23s}"

    @property
    def time_since_last_server_contact(self):
        """get time since last Server contact

        Returns:
            float: time since last Server contact
        """
        now = datetime.now().timestamp()
        delta = now - self.__dict__.get('_last_fetch_date', 0.0)
        return delta

    def _cache_expired(self):
        delta = self.time_since_last_server_contact
        return {'delta': delta, 'bool': delta > self._mp.caching}

    def _restructure(self, local_asset):
        # restructure downloaded data for easy access
        # beautiful effective python: dict comprehension :-)
        local_asset['properties'] = {
            p['name']: p for p in local_asset['properties']}
        local_asset['dataItems'] = {
            d['name']: d for d in local_asset['dataItems']}
        return local_asset

    def _set_oph_parameter(self):
        # for the oph(ts) function
        # this function uses the exect date to calculate
        # the interpolation line
        self._k = float(self.oph_parts /
                        (self._lastDataFlowDate - self._valstart_ts))
        # for the oph2(ts) function
        # this function uses the myplant reported hours and the
        # request time to calculate the inperpolation
        # for low validation oph this gives more consistent results
        self._k2 = float(self.oph_parts /
                         (arrow.now().timestamp() - self._valstart_ts))

    def oph(self, ts):
        """Interpolated Operating hours

        Args:
            ts (log int): timestamp

        Returns:
            float: Operating time rel. to Validation start

        doctest:
        >>> e = dmyplant2.Engine(mp, eng)
        >>> 6160.0 <= e.oph2(arrow.get("2021-03-01 00:00").timestamp()) <= 6170.0
        True
        """
        y = self._k * (ts - self._valstart_ts)
        y = y if y > 0.0 else 0.0
        return y

    def oph2(self, ts):
        """Interpolated Operating hours, method 2

        Args:
            ts (log int): timestamp

        Returns:
            float: Operating time rel. to Validation start

        doctest:
        >>> e = dmyplant2.Engine(mp, eng)
        >>> 6160.0 <= e.oph2(arrow.get("2021-03-01 00:00").timestamp()) <= 6170.0
        True
        """
        y = self._k2 * (ts - self._valstart_ts)
        y = y if y > 0.0 else 0.0
        return y

    def _engine_data(self, eng) -> dict:
        # extract and store important data
        def calc_values(d) -> dict:
            oph_parts = float(d['Count_OpHour']) - float(d['oph@start'])
            d.update({'oph parts': oph_parts})
            return d

        dd = {}
        from_asset = {
            'nokey': ['serialNumber', 'status', 'id', 'model'],
            'properties': ['Engine Version', 'Engine Type', 'Engine Series', 'IB Unit Commissioning Date', 'Design Number',
                        'Engine ID', 'IB Control Software', 'IB Item Description Engine', 'IB Project Name'],
            'dataItems': ['Count_OpHour', 'Count_Start']}

        for key in from_asset:
            for ditem in from_asset[key]:
                dd[ditem] = self.get_data(key, ditem)

        dd['Name'] = eng['Validation Engine']
        self.Name = eng['Validation Engine']

        if dd['Engine Type']:
            dd['P'] = int(str(dd['Engine Type'])[-2:])
            self._P = dd['P']
        else:
            raise Exception(f'Key "Engine Type" missing in asset of SN {self._sn}\nconsider a patch in patch.json')

        dd['val start'] = eng['val start']
        dd['oph@start'] = eng['oph@start']

        # add calculated items
        dd = calc_values(dd)
        self._valstart_ts = epoch_ts(arrow.get(dd['val start']).timestamp())
        self._lastDataFlowDate = epoch_ts(dd['status'].get(
            'lastDataFlowDate', None))

        return dd

    def _save(self):
        try:
            self._info['last_fetch_date'] = self._last_fetch_date
            self._info['Validation Engine'] = self._d['IB Project Name']
            self._info['val start'] = arrow.get(
                self._eng['val start']).format('YYYY-MM-DD')
            with open(self._infofile, 'w') as f:
                json.dump(self._info, f)
        except FileNotFoundError:
            errortext = f'File {self._infofile} not found.'
            logging.error(errortext)

        try:
            with open(self._picklefile, 'wb') as handle:
                pickle.dump(self.__dict__, handle, protocol=4)
        except FileNotFoundError:
            errortext = f'File {self._picklefile} not found.'
            logging.error(errortext)
            # raise Exception(errortext)

    def get_data(self, key, item):
        """
        Get Item Value by Key, Item Name pair
        valid Myplant Keys are
        'nokey' data Item in Asset Date base structure
        'properties' data Item is in 'properties' list
        'dataItems' data Item is in 'dataItems' list

        e.g.: oph = e.get_data('dataItems','Count_OpHour')
        
        doctest:
        >>> e = dmyplant2.Engine(mp, eng)
        >>> e.get_data('nokey','id')
        117617
        >>> e.get_data('nokey','nothing') == None
        True
        >>> e.get_data('dataItems','Power_PowerNominal')
        4500.0
        >>> e.get_data('dataItems','nothing') == None
        True
        >>> e.get_data('properties','Engine ID')
        'M4'
        >>> e.get_data('properties','nothing') == None
        True
        """
        return self.asset.get(item, None) if key == 'nokey' else self.asset[key].setdefault(item, {'value': None})['value']

    def get_property(self, item):
        """
        Get properties Item Value by Item Name

        e.g.: vers = e.get_property("Engine Version")

        doctest:
        >>> e = dmyplant2.Engine(mp, eng)
        >>> e.get_property('Engine ID')
        'M4'
        >>> e.get_property('nothing') == None
        True
        """
        return self.get_data('properties', item)

    def get_dataItem(self, item):
        """
        Get  dataItems Item Value by Item Name

        e.g.: vers = e.get_dataItem("Monic_VoltCyl01")

        doctest:
        >>> e = dmyplant2.Engine(mp, eng)
        >>> e.get_dataItem('Power_PowerNominal')
        4500.0
        >>> e.get_dataItem('nothing') == None
        True
        """
        return self.get_data('dataItems', item)

    def historical_dataItem(self, itemId, timestamp):
        """
        Get historical dataItem
        dataItemId  int64   Id of the DataItem to query.
        timestamp   int64   Optional,  timestamp in the DataItem history to query for.

        doctest:
        >>> e = dmyplant2.Engine(mp, eng)
        >>> e.historical_dataItem(161, arrow.get("2021-03-01 00:00").timestamp())
        12575.0
        """
        try:
            res = self._mp.historical_dataItem(
                self.id, itemId, mp_ts(timestamp)).get('value', None)
        except Exception as e:
            print(e)
            res = None
        return res

    def batch_hist_dataItem(self, itemId, p_from, p_to, timeCycle=3600):
        """
        Get np.array of a single dataItem history
        dataItemId  int64   Id of the DataItem to query.
        p_from      int64   timestamp start timestamp.
        p_to        int64   timestamp stop timestamp.
        timeCycle   int64   interval in seconds.
        """
        try:
            res = self._mp.history_dataItem(
                self.id, itemId, mp_ts(p_from), mp_ts(p_to), timeCycle)
            df = pd.DataFrame(res)
            df.columns = ['timestamp', str(itemId)]
            return df
        except:
            pass

    def hist_data(self, itemIds={161: ['CountOph', 'h']}, p_limit=None, p_from=None, p_to=None, timeCycle=86400,
                  assetType='J-Engine', includeMinMax='false', forceDownSampling='false', slot=0, debug=False):
        """
        Get pandas dataFrame of dataItems history, either limit or From & to are required
        ItemIds             dict   e.g. {161: ['CountOph','h']}, dict of dataItems to query.
        p_limit             number of datapoints back from "now".
        p_from              string from iso date or timestamp,
        p_to                string stop iso date or timestamp.
        timeCycle           int64  interval in seconds.
        assetType           string default 'J-Engine'
        includeMinMax       string 'false'
        forceDownSampling   string 'false'
        slot                int     dataset differentiator, defaults to 0
        """

        def collect_info():
            # File Metadata
            info = self._info
            info['p_from'] = p_from
            info['p_to'] = p_to
            info['Timezone'] = 'Europe/Vienna'
            info['timeCycle'] = timeCycle
            info['Exported_By'] = self._mp.username
            info['Export_Date'] = arrow.now().to(
                'Europe/Vienna').format('DD.MM.YYYY - HH:mm')
            info['dataItems'] = itemIds
            return pd.DataFrame.from_dict(info)

        def check_and_loadfile(p_from, fn):
            ldf = pd.DataFrame([])
            last_p_to = p_from
            if os.path.exists(fn):
                try:
                    dinfo = pd.read_hdf(fn, "info").to_dict()
                    # wenn die daten im file den angeforderten daten entsprechen ...
                    if set(itemIds) == set(dinfo['dataItems']):
                        ffrom = list(dinfo['p_from'].values())[0]
                        if ffrom.to('Europe/Vienna') <= p_from.to('Europe/Vienna'):
                            ldf = pd.read_hdf(fn, "data")
                            os.remove(fn)
                            # Check last lp_to in the file and update the file ....
                            last_p_to = arrow.get(
                                list(ldf['time'][-2:-1])[0]).to('Europe/Vienna')
                            # list(ldf['time'][-2:-1])[0] + timeCycle)
                            # new starting point ...
                            if debug:
                                print(f"\nitemIds: {set(itemIds)}, Shape={ldf.shape}, from: {p_from.format('DD.MM.YYYY - HH:mm')}, to:   {last_p_to.format('DD.MM.YYYY - HH:mm')}, loaded from {fn}")
                except:
                    pass
            return ldf, last_p_to

        try:
            df = pd.DataFrame([])
            # fn = fr"./data/{self._sn}_{p_from.timestamp}_{timeCycle}_{slot:02d}.hdf"
            fn = fr"./data/{self._sn}_{timeCycle}_{slot:02d}.hdf"
            df, np_from = check_and_loadfile(p_from, fn)

            np_to = arrow.get(p_to).shift(seconds=-timeCycle)
            if np_from.to('Europe/Vienna') < np_to.to('Europe/Vienna'):
                ndf = self._mp.hist_data(
                    self.id, itemIds, np_from, p_to, timeCycle)
                df = df.append(ndf)
                if debug:
                    print(f"\nitemIds: {set(itemIds)}, Shape={ndf.shape}, from: {np_from.format('DD.MM.YYYY - HH:mm')}, to:   {p_to.format('DD.MM.YYYY - HH:mm')}, added to {fn}")

            df.reset_index(drop=True, inplace=True)

            dinfo = collect_info()
            dinfo.to_hdf(fn, "info", complevel=6)
            df.to_hdf(fn, "data", complevel=6)

            return df
        except:
            raise ValueError("Engine hist_data Error")

    def scan_for_highres_DataFrames(self, dat):
        df = pd.DataFrame([])
        alarms = self.batch_hist_alarms(
            p_from=arrow.get(self.val_start).to('Europe/Vienna'),
            p_to=arrow.get(self.val_start).to('Europe/Vienna').shift(months=3)
        )
        alarms = alarms[(alarms['name'] == '1232') |
                        (alarms['name'] == '1231')]
        for i, row in enumerate(alarms[['name', 'message', 'datetime']][::-1].values):
            print(row)
            df = df.append(
                self.hist_data(
                    dat,
                    p_from=arrow.get(
                        row[2], 'MM-DD-YYYY HH:mm').shift(minutes=-10),
                    p_to=arrow.get(
                        row[2], 'MM-DD-YYYY HH:mm').shift(minutes=10),
                    timeCycle=1,
                    slot=i+1))
        return df

    def _batch_hist_dataItems(self, itemIds={161: ['CountOph', 'h']}, p_limit=None, p_from=None, p_to=None, timeCycle=3600,
                              assetType='J-Engine', includeMinMax='false', forceDownSampling='false'):
        """
        Get pandas dataFrame of dataItems history, either limit or From & to are required
        ItemIds             dict   e.g. {161: ['CountOph','h']}, dict of dataItems to query.
        limit               int64, number of points to download
        p_from              string from iso date or timestamp,
        p_to                string stop iso date or timestamp.
        timeCycle           int64  interval in seconds.
        assetType           string default 'J-Engine'
        includeMinMax       string 'false'
        forceDownSampling   string 'false'
        """
        try:
            tt = r""
            if p_limit:
                tt = r"&limit=" + str(p_limit)
            else:
                if p_from and p_to:
                    tt = r'&from=' + str(int(arrow.get(p_from).timestamp()) * 1000) + \
                        r'&to=' + str(int(arrow.get(p_to).timestamp()) * 1000)
                else:
                    raise Exception(
                        r"batch_hist_dataItems, invalid Parameters")

            tdef = itemIds
            tdj = ','.join([str(s) for s in tdef.keys()])

            ttimecycle = timeCycle
            tassetType = assetType
            tincludeMinMax = includeMinMax
            tforceDownSampling = forceDownSampling

            url = r'/asset/' + str(self.id) + \
                r'/history/batchdata' + \
                r'?assetType=' + str(tassetType) + \
                tt + \
                r'&dataItemIds=' + str(tdj) + \
                r'&timeCycle=' + str(ttimecycle) + \
                r'&includeMinMax=' + str(tincludeMinMax) + \
                r'&forceDownSampling=' + str(tforceDownSampling)

            # fetch data from myplant ....
            data = self._mp.fetchdata(url)

            # restructure data to dict
            ds = dict()
            ds['labels'] = ['time'] + [tdef[x][0] for x in data['columns'][1]]
            ds['data'] = [[r[0]] + [rr[0] for rr in r[1]]
                          for r in data['data']]

            # import to Pandas DataFrame
            df = pd.DataFrame(ds['data'], columns=ds['labels'])
            # Addtional Datetime column calculated from timestamp
            df['datetime'] = pd.to_datetime(df['time'] * 1000000)
            return df
        except:
            raise Exception("Error in call to _batch_hist_dataItems")

    def _Validation_period_LOC_prelim(self):
        """ Work in progress on a better LOC Function
            
            - synchronize other data etc.

        """
        def _localfunc(dloc):
            dat0 = {
                161: ['Count_OpHour', 'h'], 
                102: ['Power_PowerAct', 'kW'],
                228: ['Hyd_OilCount_Trend_OilVolume','ml'],
                107: ['Various_Values_SpeedAct','rpm'],
                69: ['Hyd_PressCoolWat','bar'],
                16546: ['Hyd_PressOilDif','bar']
            }

            l_from = arrow.get(dloc.datetime.iloc[-1])
            _cyclic = self.hist_data(
                itemIds= dat0, 
                p_from = l_from,
                p_to=arrow.now('Europe/Vienna'),
                timeCycle=60,
                slot=11
            )

            ts_list = list(dloc['time'])
            loc_list = list(dloc['OilConsumption'])

            # Add Values from _cyclic to dloc
            # add Count_OpHour
            #value_list = [_cyclic['Count_OpHour'].iloc[_cyclic['time'].values.searchsorted(a)] - self.oph_start for a in ts_list]
            #dloc['oph_parts'] = value_list
            
            # add Count_OpHour
            #value_list = [_cyclic['Power_PowerAct'].iloc[_cyclic['time'].values.searchsorted(a)] for a in ts_list]
            #dloc['Power_PowerAct'] = value_list

            # add Count_OpHour
            #value_list = [_cyclic['Hyd_OilCount_Trend_OilVolume'].iloc[_cyclic['time'].values.searchsorted(a)] for a in ts_list]
            #dloc['Hyd_OilCount_Trend_OilVolume'] = value_list


            # Add Values from dloc to _cyclic
            #_cyclic['OilConsumption'] = np.nan
            #for i, ts in enumerate(ts_list):
            #    _cyclic['OilConsumption'].iloc[_cyclic['time'].values.searchsorted(ts)] = loc_list[i]
                #print(f"LOC {loc_list[i]} at position {ts} inserted.")

            return dloc, _cyclic

        dloc = self.Validation_period_LOC()
        dloc=_localfunc(dloc)
        return dloc

    def Validation_period_LOC(self):
        """Oilconsumption vs. Validation period

        Raises:
            Exception: [description]
            Exception: [description]

        Returns:
            pd.DataFrame:

            columns
            227: ['OilConsumption', 'g/kWh'],
            237: ['DeltaOpH', 'h'],
            228: ['OilVolume', 'ml'],
            225: ['ActiveEnergy', 'MWh'],
            226: ['AvgPower', 'kW']
        """
        # Lube Oil Consumption data
        locdef = {
            227: ['OilConsumption', 'g/kWh'],
            # 237: ['DeltaOpH', 'h'],
            # 228: ['OilVolume', 'ml'],
            # 225: ['ActiveEnergy', 'MWh'],
            226: ['AvgPower', 'kW'],
        }

        limit = 4000

        try:
            dloc = self._batch_hist_dataItems(
                itemIds=locdef, p_limit=limit, timeCycle=1)
            #dloc = add_column(dloc, 161)
            cnt = dloc['OilConsumption'].count()
            DebugStr = f"Data Start {arrow.get(dloc.datetime.iloc[-1]).format('DD.MM.YYYY')}\nVal  Start {arrow.get(self.val_start).format('DD.MM.YYYY')}"
            DebugStr = "LOC, all available data received,\n" + DebugStr if (cnt != limit) else f"limit={int(limit)},\n" + DebugStr
            print(DebugStr)
        except:
            raise Exception("Loop Error in Validation_period_LOC")

        # skip values before validation start
        dloc = dloc[dloc.datetime > pd.to_datetime(self.val_start)]
        
        # Filter outliers by < 3 * stdev - remove refilling, engine work etc..
        dloc = dloc[np.abs(dloc.OilConsumption-dloc.OilConsumption.mean())
                    <= (3*dloc.OilConsumption.std())]

        # Calculate Rolling Mean values for Power and LOC
        dloc['LOC'] = dloc.OilConsumption.rolling(10).mean()
        dloc['Pow'] = dloc.AvgPower.rolling(10).mean()
        return dloc

    def timestamp_LOC(self,starttime, endtime, windowsize=50, return_OPH=False):  #starttime, endtime, 
        """Oilconsumption vs. Validation period

        Args:
            starttime: arrow object in right timezone
            endtime: arrow object in right timezone
            windowsize (optional): Engine instance to get number of cylinders from
            return_OPH (optional): Option to directly return the engine OPH in the dataframe at the LOC-data points

        Returns:
            pd.DataFrame:

        """
        #Lube Oil Consumption data
        locdef = ['Operating hours engine', 'Oil counter active energy', 'Oil counter power average', 'Oil counter oil consumption', 'Oil counter oil volume', 'Oil counter operational hours delta']
        
        ans1=datastr_to_dict(locdef)
        locdef=ans1[0]
        try:
            dloc = self.hist_data(
                itemIds=locdef, p_from=starttime,
                p_to=endtime, timeCycle=3600, slot=1)
            dloc.rename(columns = ans1[1], inplace = True)

            dloc.drop(['time'], axis=1, inplace=True)
            dloc = dloc.set_index('datetime')
            dloc=dloc.drop_duplicates(['Oil counter active energy', 'Oil counter power average', 'Oil counter oil consumption', 'Oil counter oil volume', 'Oil counter operational hours delta'])


            dloc.drop(dloc[((dloc['Oil counter oil volume']*10)%1!=0)].index, inplace=True)
            dloc.drop(dloc[(dloc['Oil counter power average']%1!=0)].index, inplace=True)
            dloc.drop(dloc[(dloc['Oil counter operational hours delta']%1!=0)].index, inplace=True)


            hoursum = 0
            volumesum = 0
            energysum = 0
            powersum = 0
            count = 0

            LOC_ws = []
            LOC_raw = []
            hours_filtered=[]
            OPH_engine=[]

            for i in range(len(dloc)):
                hoursum = hoursum + dloc.iloc[i, dloc.columns.get_loc('Oil counter operational hours delta')]
                volumesum = volumesum + dloc.iloc[i, dloc.columns.get_loc('Oil counter oil volume')]
                energysum = energysum + dloc.iloc[i, dloc.columns.get_loc('Oil counter active energy')]

                if hoursum >= windowsize:
                    LOC_ws.append(volumesum * 0.886 / energysum) #only make 3 decimal points
                    hoursum = 0
                    volumesum = 0
                    energysum = 0
                else:
                    LOC_ws.append(np.nan)

                LOC_raw.append (dloc.iloc[i, dloc.columns.get_loc('Oil counter oil consumption')])
                OPH_engine.append(dloc.iloc[i, dloc.columns.get_loc('Operating hours engine')])
                hours_filtered.append(dloc.index[i])

            
            if return_OPH:
                dfres = pd.DataFrame(data={'datetime': hours_filtered, 'OPH_engine': OPH_engine, 'LOC_average': LOC_ws, 'LOC_raw': LOC_raw})
            else:
                dfres = pd.DataFrame(data={'datetime': hours_filtered, 'LOC_average': LOC_ws, 'LOC_raw': LOC_raw})

            dfres=dfres.set_index('datetime')
        except:
                raise Exception("Loop Error in Validation_period_LOC")
        return dfres

    def batch_hist_alarms(self, p_severities=[500, 600, 650, 700, 800], p_offset=0, p_limit=None, p_from=None, p_to=None):
        """
        Get pandas dataFrame of Events history, either limit or From & to are required
        p_severities        list
                                500,600,650 ... operational messages
                                700         ... warnings
                                800         ... alarms
        p_offset            int64, number of messages to skip
        p_limit             int64, number of messages to download
        p_from              string timestamp in milliseconds.
        p_to                string timestamp in milliseconds.
        """
        try:
            tt = r""
            if p_limit:
                tt = r"&offset=" + str(p_offset) + \
                    r"&limit=" + str(p_limit)
            else:
                if p_from and p_to:
                    tt = r'&from=' + str(int(arrow.get(p_from).timestamp()) * 1000) + \
                        r'&to=' + str(int(arrow.get(p_to).timestamp()) * 1000)
                else:
                    raise Exception(
                        r"batch_hist_alarms, invalid Parameters")

            tsvj = ','.join([str(s) for s in p_severities])

            url = r'/asset/' + str(self.id) + \
                r'/history/alarms' + \
                r'?severities=' + str(tsvj) + tt

            # fetch messages from myplant ....
            messages = self._mp.fetchdata(url)

            # import to Pandas DataFrame
            dm = pd.DataFrame(messages)
            # Addtional Datetime column calculated from timestamp
            dm['datetime'] = pd.to_datetime(
                dm['timestamp'] * 1000000.0).dt.strftime("%m-%d-%Y %H:%m")
            return dm
        except:
            raise

    @ property
    def id(self):
        """
        MyPlant Asset id

        e.g.: id = e.id
        """
        return self.get_data('nokey', 'id')
        # return self._d['id']

    @ property
    def serialNumber(self):
        """
        MyPlant serialNumber
        e.g.: serialNumber = e.serialNumber
        """
        return self.get_data('nokey', 'serialNumber')
        # return self._d['serialNumber']

    @ property
    def oph_parts(self):
        """
        Oph since Validation Start
        """
        return int(self.Count_OpHour - self.oph_start)

    @ property
    def properties(self):
        """
        properties dict
        e.g.: prop = e.properties
        """
        return self.asset['properties']

    @ property
    def dataItems(self):
        """
        dataItems dict
        e.g.: dataItems = e.dataItems
        """
        return self.asset['dataItems']

    @ property
    def val_start(self):
        """
        Individual Validation Start Date
        as String
        """
        return str(self._eng['val start'])
        # return self._valstart_ts

    @ property
    def valstart_ts(self):
        """
        Individual Validation Start Date
        as EPOCH timestamp
        e.g.: vs = e.valstart_ts
        """
        return epoch_ts(self._eng['val start'].timestamp())

    @ property
    def oph_start(self):
        """
        oph at Validation Start
        as Int
        """
        return int(self._eng['oph@start'])

    @ property
    def Count_OpHour(self):
        """
        get current OP Hours
        """
        return int(self.get_dataItem('Count_OpHour'))

    ############################
    #Calculated & exposed values
    ############################

    @ staticmethod
    def _bore(platform):
        """
        return bore for platform in [mm]
        """
        lbore = {
            '9': 310.0,
            '6': 190.0,
            '4': 145.0,
            '3': 135.0
        }
        return lbore[platform]

    @ property
    def bore(self):
        """
        bore in [mm]
        """
        lkey = self.get_property('Engine Series')
        return self._bore(lkey)

    @ staticmethod
    def _stroke(platform):
        """
        return stroke for platform in [mm]
        """
        lstroke = {
            '9': 350.0,
            '6': 220.0,
            '4': 185.0,
            '3': 170.0
        }
        return lstroke[platform]

    @ property
    def stroke(self):
        """
        stroke in [mm]
        """
        lkey = self.get_property('Engine Series')
        return self._stroke(lkey)

    @ classmethod
    def _cylvol(cls, platform):
        """
        Swept Volume for platform per Cylinder in [l]
        """
        lbore = cls._bore(platform)
        lstroke = cls._stroke(platform)
        return (lbore / 100.0) * (lbore / 100.0) * np.pi / 4.0 * (lstroke / 100.0)

    @ classmethod
    def _mechpower(cls, platform, cylanz, bmep, speed):
        """
        mechanical power in [kW]
        platform ... '3','4','6','9'
        cylanz ... int
        bmep ... bar
        speed ... int
        """
        return np.around(cls._cylvol(platform) * cylanz * bmep * speed / 1200.0, decimals=0)

    @ property
    def cylvol(self):
        """
        Swept Volume per Cylinder in [l]
        """
        lkey = self.get_property('Engine Series')
        return self._cylvol(lkey)

    @ property
    def engvol(self):
        """
        Swept Volume per Engine in [l]
        """
        lkey = self.get_property('Engine Series')
        if not lkey:
            lkey = '6'
        return self._cylvol(lkey) * self.Cylinders

    @ property
    def Cylinders(self):
        """
        Number of Cylinders
        """
        return int(str(self.get_property('Engine Type')[-2:]))

    @ property
    def P_nominal(self):
        """
        Nominal electrical Power in [kW]
        """
        return np.around(float(self.get_dataItem('Power_PowerNominal')), decimals=0)

    @ property
    def cos_phi(self):
        """
        cos phi ... current Power Factor[-]
        """
        return self.get_dataItem('halio_power_fact_cos_phi')

    @ property
    def Generator_Efficiency(self):
        # gmodel = self.get_property('Generator Model')
        # cosphi = self.get_dataItem('halio')
        el_eff = {
            '624': 0.981,
            '620': 0.98,
            '616': 0.976,
            '612': 0.986,
            '312': 0.965,
            '316': 0.925,
            '320': 0.975,
            '412': 0.973,
            '416': 0.974,
            '420': 0.973
        }
        lkey = self.get_property('Engine Type')
        return el_eff[lkey] or 0.95

    @ property
    def Pmech_nominal(self):
        """
        Nominal, Calculated mechanical Power in [kW]
        """
        return np.around(self.P_nominal / self.Generator_Efficiency, decimals=1)

    @ property
    def Speed_nominal(self):
        """
        Nominal Speed in [rp/m]
        """
        speed = {
            '3': 1500.0,
            '4': 1500.0,
            '5': 1500.0,
            '6': 1500.0,
            '9': 1000.0
        }
        lkey = self.get_property('Engine Type')[:1]
        return speed[lkey]
        # return self.get_dataItem('Para_Speed_Nominal')

    @ property
    def BMEP(self):
        return np.around(1200.0 * self.Pmech_nominal / (self.engvol * self.Speed_nominal), decimals=1)

    @ property
    def dash(self):
        _dash = dict()
        _dash['Name'] = self.Name
        _dash['Engine ID'] = self.get_property('Engine ID')
        _dash['Design Number'] = self.get_property('Design Number')
        _dash['Engine Type'] = self.get_property('Engine Type')
        _dash['Engine Version'] = self.get_property('Engine Version')
        _dash['P'] = self.Cylinders
        _dash['P_nom'] = self.Pmech_nominal
        _dash['BMEP'] = self.BMEP
        _dash['serialNumber'] = self.serialNumber
        _dash['id'] = self.id
        _dash['Count_OpHour'] = self.Count_OpHour
        _dash['val start'] = pd.to_datetime(self.val_start, format='%Y-%m-%d')
        _dash['oph@start'] = self.oph_start
        _dash['oph parts'] = self.oph_parts
        _dash['LOC'] = self.get_dataItem(
            'RMD_ListBuffMAvgOilConsume_OilConsumption')
        return _dash


class Engine_SN(Engine):
    """
    Engine Object with serialNumber constructor
    inherited from Engine

    doctest:
    >>> e = dmyplant2.Engine_SN(mp,'1320072') 
    >>> print(f"{e._sn}")
    1320072
    """

    def __init__(self, mp, sn):
        """Constructor Init

        Args:
            mp (dmyplant2.maplant instance): Myplant Access Function Class
            sn (string): serialNumber
        """
        eng = {}
        fname = os.getcwd() + '/data/' + str(sn)
        self._infofile = fname + '.json'

        # load info json & lstfetchdate
        try:
            with open(self._infofile) as f:
                eng = json.load(f)
                eng['serialNumber'] = int(sn)
                # if no 'val start' in json, fake it ...
                if not 'val start' in eng:
                    eng['val start'] = '2000-01-01'
                # if no 'n' in json, fake it ...
                if not 'n' in eng:
                    eng['n'] = 0
        except FileNotFoundError:
            # minimal info record to allow myplant
            # to fetch data for a never conatacted engine
            eng = {
                'n': 0,
                'serialNumber': int(sn),
                'Validation Engine': 'fake Name',
                'val start': '2000-01-01',
                'oph@start': 0
            }

        super().__init__(mp, eng)

        # use Myplant Data to update some fake variables
        self.Name = self._d['IB Project Name']
        self._eng['Validation Engine'] = self.Name

if __name__ == "__main__":

    import dmyplant2
    import pandas
    eng = {
    'n': 16,
    'Validation Engine': 'BMW LANDSHUT 4.10',
    'serialNumber': 1320072,
    'val start': '2020-02-07',
    'oph@start': 6316,
    'starts@start': 768.0,
    'Asset ID': 117617.0,
    'Old PU first replaced OPH': 1742.0,
    'Old PUs replaced before upgrade': 4.0
    }

    dmyplant2.cred()
    mp = dmyplant2.MyPlant(0)
    #vl = dmyplant2.Validation(mp, dval, cui_log=False)

    import doctest
    doctest.testmod()

    # >>> eng = {'n': 16, \
    # 'Validation Engine': 'BMW LANDSHUT 4.10', \
    # 'serialNumber': 1320072, \
    # 'val start': '2020-02-07', \
    # 'oph@start': 6316, \
    # 'starts@start': 768.0 }
