from datetime import datetime
from functools import reduce
import pandas as pd
import numpy as np
import logging
from dmyplant2.dEngine import Engine
from pprint import pprint as pp
from scipy.stats.distributions import chi2


class HandleID():
    df = None
    def __init__(self, filename=None, datdict=None):
        if filename:
            self._load_csv(filename)
        elif datdict:
            self._load_dict(datdict)
        else:
            raise ValueError("no Request data defined")        

    def _load_csv(self, filename):
        self.df = pd.read_csv(filename, sep=';', encoding='utf-8')

    def _load_dict(self, dat):
        self.df = pd.DataFrame(
            [[k]+v for k, v in dat.items()], columns=['ID', 'myPlantName', 'unit'])

    def _unit_name(self, name):
        try:
            ret = list(self.df[self.df['myPlantName'] == name]['unit'])[0]
        except:
            raise ValueError(f"HandleID: ItemId Name '{name}' not found")
        return ret

    def _unit_id(self, id):
        try:
            ret = list(self.df[self.df['ID'] == id]['unit'])[0]
        except:
            raise ValueError(f"HandleID: ItemId number '{id}' not found")        
        return ret

    def datdict(self):
        return {rec['ID']: [rec['myPlantName'], rec['unit']] for rec in self.df.to_dict('records')}

    def unit(self, id=None, name=None):
        if id:
            return self._unit_id(id)
        elif name:
            return self._unit_name(name)
        else:
            raise ValueError("no valid Parameters provided (id or name")


class Validation:

    _dash = None
    _val = None
    _engines = []

    def __init__(self, mp, dval, lengine=Engine, eval_date=None, cui_log=False):
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
            try:
                e = lengine(mp, eng)
            except:
                raise Exception("Engine Instance could not be created.")
            self._engines.append(e)
            log = f"{eng['n']:02d} {e}"
            logging.info(log)
            if cui_log:
                print(log)

        # create dashboard with list comprehension
        ldash = [e.dash for e in self._engines]
        # dashboard as pandas Dataframe
        self._dash = pd.DataFrame(ldash)

    @ classmethod
    def load_def_csv(cls, filename):
        """load CSV Validation definition file 

        example content:
        n;Validation Engine;serialNumber;val start;oph@start;starts@start;Asset ID;Old PU first replaced OPH;Old PUs replaced before upgrade
        0;POLYNT - 2 (1145166-T241) --> Sept;1145166;12.10.2020;31291;378;103791;;
        ....

        Args:
            filename ([string]): [Filename of definition file]

        Returns:
            [pd.dataFrame]: [Validation definition as dataFrame]
        """
        dv = pd.read_csv(filename, sep=';', encoding='utf-8')
        dv['val start'] = pd.to_datetime(dv['val start'], format='%d.%m.%Y')
        return dv

    @ classmethod
    def load_def_excel(cls, filename, sheetname):
        """load CSV Validation definition file 

        example content:
        n;Validation Engine;serialNumber;val start;oph@start;starts@start;Asset ID;Old PU first replaced OPH;Old PUs replaced before upgrade
        0;POLYNT - 2 (1145166-T241) --> Sept;1145166;12.10.2020;31291;378;103791;;
        ....
        
        Args:
            filename ([string]): [Filename of definition file] must include .xslx at the end
            sheetname ([string]): Relevant sheetname in file

        Returns:
            [pd.dataFrame]: [Validation definition as dataFrame]
        """

        dval=pd.read_excel(filename, sheet_name=sheetname, usecols=['Validation Engine', 'serialNumber', 'val start', 'oph@start', 'starts@start'])
        dval.dropna(inplace=True)
        dval['n']=dval.index #add column 'n for handling in further methods
        dval['serialNumber'] = dval['serialNumber'].astype(int).astype(str)
        print(dval)
        return dval

    @ classmethod
    def load_failures_csv(cls, filename):
        """load CSV Failure Observation file 

        example content:
        date;failures;serialNumber;comment
        28.12.2020;1;1319151;München V008 M1 Z8 - Reiber, mit Boroskop am 28.12.2020 festgestellt, Cold Scuff, Motor lief 431 Stunden nach BSI
        ....

        Args:
            filename ([string]): [Filename of Failure Observation file]

        Returns:
            [pd.dataFrame]: [Failure Observations as dataFrame]
        """
        fl = pd.read_csv(filename, sep=';', encoding='utf-8')
        fl['date'] = pd.to_datetime(fl['date'], format='%d.%m.%Y')
        return fl

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
        try:
            return [e for e in self._engines if name in e.Name]
        except:
            raise ValueError(f'Engine {name} not found in Validation Engines')

    def eng_serialNumber(self, serialNumber):
        """
        Return the Engines containing Name Validation
        """
        try:
            return [e for e in self._engines if str(serialNumber) == str(e.serialNumber)][0]
        except:
            raise ValueError(
                f'Engine SN {serialNumber} not found in Validation Engines')
