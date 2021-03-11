from datetime import datetime
from functools import reduce
import pandas as pd
import numpy as np
import logging
from dmyplant2.dEngine import Engine
from pprint import pprint as pp
from scipy.stats.distributions import chi2


class Validation:

    # define dashboard columns in expected order
    # _dashcols = [
    #     'Name',
    #     'Engine ID',
    #     'Design Number',
    #     'Engine Type',
    #     'Engine Version',
    #     'P',
    #     # 'P_NOM',
    #     # 'BMEP',
    #     'serialNumber',
    #     'id',
    #     'Count_OpHour',
    #     'val start',
    #     'oph@start',
    #     'oph parts',
    # ]
    _dash = None
    _val = None
    _engines = []

    def __init__(self, mp, dval, eval_date=None, cui_log=False):
        """ Myplant Validation object
            collects and provides the engines list.
            compiles a dashboard as pandas DataFrame
            dval ... Pandas DataFrame with the Validation Definition,
                     defined in Excel sheet 'validation'
        """
        self._mp = mp
        self._val = dval
        self._now_ts = datetime.now().timestamp()
        self._eval_ts = self._now_ts if not eval_date else eval_date
        self._valstart_ts = dval['val start'].min()

        engines = self._val.to_dict('records')
        # create and initialise all Engine Instances
        self._engines = []
        for eng in engines:
            e = Engine(mp, eng)
            self._engines.append(e)
            log = f"{eng['n']:02d} {e}"
            logging.info(log)
            if cui_log:
                print(log)

        # iterate over engines and columns
        #ldash = [[e._d[c] for c in self._dashcols] for e in self._engines]
        ldash = [e.dash for e in self._engines]
        # dashboard as pandas Dataframe
        #self._dash = pd.DataFrame(ldash, columns=self._dashcols)
        self._dash = pd.DataFrame(ldash)

    @ property
    def now_ts(self):
        """the current date as EPOCH timestamp"""
        return self._now_ts

    @ property
    def eval_ts(self):
        """the current date as EPOCH timestamp"""
        return self._eval_ts

    @ property
    def valstart_ts(self):
        """Validation Start as EPOCH timestamp"""
        return self._valstart_ts.timestamp()

    # @ property
    # def valstart(self):
    #     return self._valstart_ts

    @ property
    def dashboard(self):
        """ Validation Dasboard as Pandas DataFrame """
        return self._dash

    @ property
    def properties_keys(self):
        """
        Properties: Collect all Keys from all Validation engines
        in a list - remove double entries
        """
        keys = []
        for e in self._engines:
            keys += e.properties.keys()     # add keys of each engine
            keys = list(set(keys))          # remove all double entries
        keys = sorted(keys, key=str.lower)
        dd = []
        for k in keys:                      # for all keys in all Val Engines
            for e in self._engines:         # iterate through al engines
                if k in e.properties.keys():
                    d = e.properties.get(k, None)  # get property dict
                    if d['value']:                 # if value exists
                        dd.append([d['name'], d['id']])  # store name, id pair
                        break
        return pd.DataFrame(dd, columns=['name', 'id'])

    @ property
    def dataItems_keys(self):
        """
        DataItems: Collect all Keys from all Validation engines
        in a list - remove double entries
        """
        keys = []
        for e in self._engines:
            keys += e.dataItems.keys()     # add keys of each engine
            keys = list(set(keys))          # remove all double entries
        keys = sorted(keys, key=str.lower)
        dd = []
        for k in keys:                      # for all keys in all Val Engines
            for e in self._engines:         # iterate through al engines
                if k in e.dataItems.keys():
                    d = e.dataItems.get(k, None)  # get dataItem dict
                    if d.get('name', None):                 # if value exists
                        dd.append([
                            d.get('name', None),
                            d.get('unit', None),
                            d.get('id', None)
                        ])
                        break
        return pd.DataFrame(dd, columns=['name', 'unit', 'id'])

    @ property
    def properties(self):
        """
        Properties: Asset Data properties of all Engines
        as Pandas DataFrame
        """
        # Collect all Keys in a big list and remove double counts
        keys = []
        for e in self._engines:
            keys += e.properties.keys()  # add keys of each engine
            keys = list(set(keys))  # remove all double entries
        keys = sorted(keys, key=str.lower)
        try:
            keys.remove('IB ItemNumber Engine')
            keys.insert(0, 'IB ItemNumber Engine')
        except ValueError:
            raise
        # Collect all values in a Pandas DateFrame
        loc = [[e.get_property(k)
                for k in keys] + [e.id, e.Name] for e in self._engines]
        return pd.DataFrame(loc, columns=keys + ['AssetID', 'Name'])

    @ property
    def dataItems(self):
        """
        dataItems: Asset Data dataItems of all Engines
        as Pandas DataFrame
        """
        # Collect all Keys in a big list and remove double counts
        keys = []
        for e in self._engines:
            keys += e.dataItems.keys()
            keys = list(set(keys))
        keys = sorted(keys, key=str.lower)
        loc = [[e.get_dataItem(k)
                for k in keys] + [e.Name] for e in self._engines]
        return pd.DataFrame(loc, columns=keys + ['Name'])

    @ property
    def validation_definition(self):
        """
        Validation Definition Information as pandas DataFrame
        """
        return self._val

    @ property
    def engines(self):
        """
        list of Validation Engine Objects
        """
        return self._engines

    def eng_name(self, name):
        """
        Return the Engines containing Name Validation
        """
        return [e for e in self._engines if name in e.Name]

    def eng_serialNumber(self, serialNumber):
        """
        Return the Engines containing Name Validation
        """
        return [e for e in self._engines if str(serialNumber) == str(e.serialNumber)][0]
