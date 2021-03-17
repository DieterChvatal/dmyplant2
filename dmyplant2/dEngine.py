from datetime import datetime, timedelta
import math
from pprint import pprint as pp
import pandas as pd
import numpy as np
from dmyplant2.dMyplant import epoch_ts, mp_ts
import sys
import os
import pickle
import logging
import json
import arrow


class Engine(object):
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

    def __init__(self, mp, eng):
        """Engine Constructor

        Args:
            mp (dmyplant2.Myplant): Myplant class instance
            eng (pd.dataFrame): Validation engine input data
        """
        # take engine Myplant Serial Number from Validation Definition
        self._mp = mp
        self._eng = eng
        self._sn = str(eng['serialNumber'])
        fname = os.getcwd() + '/data/' + self._sn
        self._picklefile = fname + '.pkl'    # load persitant data
        self._lastcontact = fname + '_lastcontact.pkl'
        try:
            with open(self._lastcontact, 'rb') as handle:
                self._last_fetch_date = pickle.load(handle)
        except:
            pass
        try:
            # fetch data from Myplant only on conditions below
            if self._cache_expired()['bool'] or (not os.path.exists(self._picklefile)):
                local_asset = self._mp.asset_data(self._sn)
                logging.debug(
                    f"{eng['Validation Engine']}, Engine Data fetched from Myplant")
                self.asset = self._restructure(local_asset)
                self._last_fetch_date = epoch_ts(datetime.now().timestamp())
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
                         (self.now_ts - self._valstart_ts))

    def oph(self, ts):
        """Interpolated Operating hours

        Args:
            ts (log int): timestamp

        Returns:
            float: Operating time rel. to Validation start
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
            'properties': ['Engine Version', 'Engine Type', 'IB Unit Commissioning Date', 'Design Number',
                           'Engine ID', 'IB Control Software', 'IB Item Description Engine', 'IB Project Name'],
            'dataItems': ['Count_OpHour', 'Count_Start']}

        for key in from_asset:
            for ditem in from_asset[key]:
                dd[ditem] = self.get_data(key, ditem)

        dd['Name'] = eng['Validation Engine']
        self.Name = eng['Validation Engine']
        dd['P'] = int(str(dd['Engine Type'])[-2:])
        self._P = dd['P']
        dd['val start'] = eng['val start']
        dd['oph@start'] = eng['oph@start']

        # add calculated items
        dd = calc_values(dd)
        self._valstart_ts = epoch_ts(dd['val start'].timestamp())
        self._lastDataFlowDate = epoch_ts(dd['status'].get(
            'lastDataFlowDate', None))
        return dd

    def _save(self):
        # pickle store object status
        try:
            with open(self._lastcontact, 'wb') as handle:
                pickle.dump(self._last_fetch_date, handle, protocol=4)
        except FileNotFoundError:
            errortext = f'File {self._lastcontact} not found.'
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

        e.g.: oph = e.get_data('dataItms','Count_OpHour')
        """
        return self.asset.get(item, None) if key == 'nokey' else self.asset[key].setdefault(item, {'value': None})['value']

    def get_property(self, item):
        """
        Get properties Item Value by Item Name

        e.g.: vers = e.get_property("Engine Version")
        """
        return self.get_data('properties', item)

    def get_dataItem(self, item):
        """
        Get  dataItems Item Value by Item Name

        e.g.: vers = e.get_dataItem("Monic_VoltCyl01")
        """
        return self.get_data('dataItems', item)

    def historical_dataItem(self, itemId, timestamp):
        """
        Get historical dataItem
        dataItemId  int64   Id of the DataItem to query.
        timestamp   int64   Optional,  timestamp in the DataItem history to query for.
        """
        try:
            res = self._mp.historical_dataItem(
                self.id, itemId, mp_ts(timestamp)).get('value', None)
        except:
            res = None
        return res

    def batch_hist_dataItem(self, itemId, p_from, p_to, timeCycle=3600):
        """
        Get np.array of dataItem history
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

    def batch_hist_dataItems(self, itemIds={161: 'CountOph'}, p_limit=None, p_from=None, p_to=None, timeCycle=86400,
                             assetType='J-Engine', includeMinMax='false', forceDownSampling='false'):
        """
        Get pandas dataFrame of dataItems history, either limit or From & to are required
        dataItemIds         dict   e.g. {161: 'CountOph'}, dict of dataItems to query.
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
                    tt = r'&from=' + str(arrow.get(p_from).timestamp * 1000) + \
                        r'&to=' + str(arrow.get(p_to).timestamp * 1000)
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
            ds['labels'] = ['time'] + [tdef[x] for x in data['columns'][1]]
            ds['data'] = [[r[0]] + [rr[0] for rr in r[1]]
                          for r in data['data']]

            # import to Pandas DataFrame
            df = pd.DataFrame(ds['data'], columns=ds['labels'])
            # Addtional Datetime column calculated from timestamp
            df['datetime'] = pd.to_datetime(df['time'] * 1000000)
            return df
        except:
            raise

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
                    tt = r'&from=' + str(arrow.get(p_from).timestamp * 1000) + \
                        r'&to=' + str(arrow.get(p_to).timestamp * 1000)
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
            '612': 0.986
        }
        lkey = self.get_property('Engine Type')
        return el_eff[lkey]

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
        return self.get_dataItem('Para_Speed_Nominal')

    @ property
    def BMEP(self):
        return np.around(1200.0 * self.Pmech_nominal / (self.engvol * self.Speed_nominal), decimals=1)

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
    def oph_start(self):
        """
        oph at Validation Start
        as Int
        """
        return int(self._eng['oph@start'])
        # return self._valstart_ts

    @ property
    def valstart_ts(self):
        """
        Individual Validation Start Date
        as EPOCH timestamp
        e.g.: vs = e.valstart_ts
        """
        return epoch_ts(self._eng['val start'].timestamp())
        # return self._valstart_ts

    @ property
    def valstart_oph(self):
        """
        Individual Validation Start Date
        as EPOCH timestamp
        e.g.: vs = e.valstart_ts
        """
        return self._d['oph@start']

    @ property
    def now_ts(self):
        """
        Actual Date & Time
        as EPOCH timestamp
        e.g.: now = e.now_ts
        """
        return datetime.now().timestamp()

    @ property
    def Count_OpHour(self):
        """
        get current OP Hours
        """
        return int(self.get_dataItem('Count_OpHour'))

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
    """

    def __init__(self, mp, sn):
        """Constructor Init

        Args:
            mp (dmyplant2.maplant instance): Myplant Access Function Class
            sn (string): serialNumber
        """
        # minimal eng record to allow myplant data fetch
        eng = {
            'serialNumber': str(sn),
            'Validation Engine': 'fake Name',
            'val start': pd.to_datetime('01.01.1970', format='%d.%m.%Y'),
            'oph@start': 0
        }
        super().__init__(mp, eng)

        # use Myplant Data to update fake variables
        self.Name = self._d['IB Project Name']
        self._eng['Validation Engine'] = self.Name
        self._eng['val start'] = pd.to_datetime(
            self._d['IB Unit Commissioning Date'], format='%Y-%m-%d')
        self._set_oph_parameter()
