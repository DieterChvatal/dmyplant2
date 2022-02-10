import arrow
import json
import base64
import requests
import logging
import os
import sys
from datetime import datetime, timedelta
from tqdm.auto import tqdm
import time
import pickle
import pandas as pd
try:
    import httplib
except:
    import http.client as httplib

maxdatapoints = 100000  # Datapoints per request, limited by Myplant

def save_json(fil, d):
    with open(fil, 'w') as f:
        json.dump(d, f)

def load_json(fil):
    with open(fil, "r", encoding='utf-8-sig') as f:
        return json.load(f)

def epoch_ts(ts) -> float:
    if ts >= 10000000000.0:
        return float(ts/1000.0)
    else:
        return float(ts)

def mp_ts(ts) -> int:
    if ts >= 10000000000.0:
        return int(ts)
    else:
        return int(ts * 1000.0)


class MyPlantException(Exception):
    pass

burl = 'https://api.myplant.io'
errortext = {
    200: 'successful operation',
    400: 'Request is missing required HTTP header \'x-seshat-token\'',
    401: 'The supplied authentication is invalid',
    403: 'No permission to access this resource',
    404: 'No data was found',
    500: 'Internal Server Error'
}


def have_internet():
    conn = httplib.HTTPConnection('api.myplant.io', timeout=5)
    try:
        conn.request("HEAD", "/")
        conn.close()
        return True
    except:
        conn.close()
        return False

class MyPlant:

    _name = ''
    _password = ''
    _session = None
    _caching = 0

    def __init__(self, caching=7200):
        """MyPlant Constructor"""
        if not have_internet():
            raise Exception("Error, Check Internet Connection!")

        self._caching = caching
        # load and manage credentials from hidden file
        try:
            with open("./data/.credentials", "r", encoding='utf-8-sig') as file:
                cred = json.load(file)
            self._name = cred['name']
            self._password = cred['password']
        except FileNotFoundError:
            raise
        if not os.path.isfile('data/dataitems.csv'):
            self.create_request_csv()

    def del_Credentials(self):
            os.remove("./data/.credentials")

    @ classmethod
    def load_dataitems_csv(cls, filename):
        """load CSV dataitems definition file

        example content:
        ID;myPlantName;unit
        52;Exhaust_TempCylAvg;C (high)
        103;Various_Values_PosTurboBypass;%
        19079;Exhaust_TCSpeedA;1000/min
        ....

        Args:
            filename (string): CSV dataitems definition file

        Returns:
            dict: CSV dataitems dict
        """
        data_req = pd.read_csv("DataItems_Request.csv",
                               sep=';', encoding='utf-8')
        dat = {a[0]: [a[1], a[2]] for a in data_req.values}
        return dat

    def deBase64(self, text):
        return base64.b64decode(text).decode('utf-8')

    def gdi(self, ds, sub_key, data_item_name):
        """Unpack value from Myplant Json datastructure based on key & DataItemName"""
        if sub_key == 'nokey':
            return ds.get(data_item_name, None)
        else:
            local = {x['value']
                     for x in ds[sub_key] if x['name'] == data_item_name}
            return local.pop() if len(local) != 0 else None

    @property
    def caching(self):
        """the current cache time"""
        return self._caching

    @property
    def username(self):
        return self.deBase64(self._name)

    def login(self):
        """Login to MyPlant"""
        if self._session is None:
            logging.debug(f"SSO {self.deBase64(self._name)} MyPlant login")
            self._session = requests.session()
            headers = {'Content-Type': 'application/json', }
            body = {
                "username": self.deBase64(self._name),
                "password": self.deBase64(self._password)
            }
            loop = 1
            try:
                while loop < 3:
                    response = self._session.post(burl + "/auth",
                                                  data=json.dumps(body), headers=headers)
                    if response.status_code == 200:
                        logging.debug(f'login {self._name} successful.')
                        break
                    else:
                        logging.error(
                            f'login failed with response code {response.status_code}')
                    loop += 1
                    logging.error(f'Myplant login attempt #{loop}')
                    time.sleep(1)
                if loop >= 3:
                    logging.error(f'Login {self.deBase64(self._name)} failed')
                    raise Exception(
                        f'Login {self.deBase64(self._name)} failed')
                    
            except:
                self.del_Credentials()
                raise Exception("Login Failed, invalid Credentials ?")
                
    def logout(self):
        """Logout from Myplant and release self._session"""
        if self._session != None:
            self._session.close()
            self._session = None

    def fetchdata(self, url):
        """login and return data based on url"""
        try:
            self.login()
            logging.debug(f'url: {url}')
            response = self._session.get(burl + url)
            if response.status_code == 200:
                logging.debug(f'fetchdata: download successful')
                res = response.json()
                return res
            else:
                logging.error(
                    f' Code: {url}, {response.status_code}, {errortext[response.status_code]}')
        except:
            raise

    def _asset_data(self, serialNumber):
        """
        Returns an Asset based on its id with all details
        including properties and DataItems.

        Parameters:
        Name	    type    Description
        sn          int     IB ItemNumber Engine
        ----------------------------------------------
        url: /asset?assetType=J-Engine&serialNumber=sn
        """
        try:
            return self.fetchdata(url=r"/asset?assetType=J-Engine&serialNumber=" + str(serialNumber))
        except:
            raise

    def historical_dataItem(self, id, itemId, timestamp):
        """
        url: /asset/{assetId}/dataitem/{dataItemId}
        Parameters:
        Name	    type    Description
        assetId     int64   Id of the Asset to query the DateItem for.
        dataItemId  int64   Id of the DataItem to query.
        timestamp   int64   Optional,  timestamp in the DataItem history to query for.
        highres     Boolean Whether to use high res data. Much slower but gives the raw data.
        """
        try:
            return self.fetchdata(url=fr"/asset/{id}/dataitem/{itemId}?timestamp={timestamp}")
        except:
            raise

    def history_dataItem(self, id, itemId, p_from, p_to, timeCycle=3600):
        """
        url: /asset/{assetId}/dataitem/{dataItemId}
        Parameters:
        Name	    type    Description
        assetId     int64   Id of the Asset to query the DateItem for.
        dataItemId  int64   Id of the DataItem to query.
        p_from      int64   timestamp start timestamp.
        p_to        int64   timestamp stop timestamp.
        timeCycle   int64   interval in seconds.
        """
        try:
            return self.fetchdata(url=fr"/asset/{id}/history/data?from={p_from}&to={p_to}&assetType=J-Engine&dataItemId={itemId}&timeCycle={timeCycle}&includeMinMax=false&forceDownSampling=false")
        except:
            raise

    def _history_batchdata(self, id, itemIds, lp_from, lp_to, timeCycle=3600):
        try:
            # comma separated string of DataItemID's
            IDS = ','.join([str(s) for s in itemIds.keys()])
            ldata = self.fetchdata(
                url=fr"/asset/{id}/history/batchdata?from={lp_from}&to={lp_to}&timeCycle={timeCycle}&assetType=J-Engine&includeMinMax=false&forceDownSampling=false&dataItemIds={IDS}")
            # restructure data to dict
            ds = dict()
            ds['labels'] = ['time'] + [itemIds[x][0]
                                       for x in ldata['columns'][1]]
            ds['data'] = [[r[0]] + [rr[0] for rr in r[1]]
                          for r in ldata['data']]
            # import data to Pandas DataFrame and return result
            df = pd.DataFrame(ds['data'], columns=ds['labels'])
            return df
        except:
            raise

    def hist_data(self, id, itemIds, p_from, p_to, timeCycle=3600
                  # , cui_log=True
                  ):
        """
        url: /asset/{assetId}/dataitem/{dataItemId}
        Parameters:
        Name	    type            Description
        assetId     int64           Id of the Asset to query the DateItem for.
        itemIds     dict            DataItem Id's, Names & Units
        p_from      int64           timestamp start timestamp in ms.
        p_to        int64           timestamp stop timestamp in ms.
        timeCycle   int64           interval in seconds.
        cui_log     boolean         report progress on CUI or not
        """
        try:
            # initialize a data collector
            df = pd.DataFrame([])

            # calculate how many full rows per request within the myplant limit are possible
            rows_per_request = maxdatapoints // len(itemIds)
            rows_total = int(p_to.timestamp() - p_from.timestamp()) // timeCycle
            pbar = tqdm(total=rows_total)

            # initialize loop
            lp_from = int(p_from.timestamp()) * 1000  # Start at lp_from
            lp_to = min((lp_from + rows_per_request * timeCycle * 1000),
                        int(p_to.timestamp()) * 1000)

            while lp_from < int(p_to.timestamp()) * 1000:
                # for now assume same itemID's are always included ... need to be included in a check
                ldf = self._history_batchdata(
                    id, itemIds, lp_from, lp_to, timeCycle)
                # and append each chunk to the return df
                df = df.append(ldf)
                pbar.update(rows_per_request)
                # calculate next cycle
                lp_from = lp_to + timeCycle * 1000
                lp_to = min((lp_to + rows_per_request *
                            timeCycle * 1000), int(p_to.timestamp()) * 1000)
            pbar.close()
            # Addtional Datetime column calculated from timestamp
            df['datetime'] = pd.to_datetime(df['time'] * 1000000)
            return df
        except:
            raise

    def stitch_df(self, **dataframes):
        """Stitch Dataframes together
        1.) Check if dataframes share the same ItemId's
        2.) Remove overlapping areas, keep the higher frequent part
        3.) return stitched Dataframe 

        Args:
            **dataframes

        Returns:
            pd.DataFrame: Stitched Dataframe
        """        
        return pd.DataFrame([])

    def create_request_csv(self):
        """Create Request_csv with id, name, unit, myPlantName and save in /data"""
        
        try:
            model=self.fetchdata('/model/J-Engine')
            dataitems=self.fetchdata('/system/localization?groups=data-items&groupResult=true')
        except:
            raise

        model=pd.json_normalize(model, record_path =['dataItems'])

        dataitems_df=pd.DataFrame(columns=['lan','dataitem', 'lan_item'])

        for lan in dataitems:
            output=pd.DataFrame(dataitems[lan]['groups'][0]['values'].items(), columns=['dataitem','myPlantName'])
            output['lan']=lan
            dataitems_df=dataitems_df.append(output, ignore_index=True)
        dataitems_df.head()

        def remove_jen (row): #with best practice could probably be shortened
            return row.split('_',1)[1]
        dataitems_df['dataitem']=dataitems_df.dataitem.apply(remove_jen)
        model=model.merge(dataitems_df[dataitems_df.lan=='en'], how='inner', left_on='name', right_on='dataitem')
        model=model.loc[:,['id', 'name', 'unit', 'myPlantName']]
        model.to_csv('data/dataitems.csv', sep=';', index=False)

    def _reshape_asset(self, rec):
        ret = dict()
        for key, value in rec.items():
            if type(value) == list:
                for lrec in value:
                    ret[lrec['name']] = lrec['value']
            else:
                ret[key] = value
        return ret

    def fetch_installed_base(self,fields, properties, dataItems, limit = None):
        url = "/asset/" + \
            "?fields=" + ','.join(fields) + \
            "&properties=" + ','.join(properties) + \
            "&dataItems="  + ','.join(dataItems) + \
            "&assetTypes=J-Engine"
        if limit:
            url = url + f"&limit={limit}"
        res = self.fetchdata(url)
        return pd.DataFrame.from_records([self._reshape_asset(a) for a in res['data']])

    def _fetch_installed_base(self):
        fields = ['serialNumber']

        properties =  [
            'Design Number','Engine Type','Engine Version','Engine Series','Engine ID',
            'Control System Type',
            'Country','IB Site Name','Commissioning Date','IB Unit Commissioning Date','Contract.Warranty Start Date', 'Contract.Warranty End Date','IB Status',
            'IB NOX', 'IB Frequency', 'IB Item Description Engine'
            ]

        dataItems = ['OperationalCondition','Module_Vers_HalIO','starts_oph_ratio','startup_counter',
        'shutdown_counter','Count_OpHour','Power_PowerNominal','Para_Speed_Nominal'
        ]
        fleet = self.fetch_installed_base(fields, properties, dataItems, limit = None)
        fleet.to_pickle('./data/Installed_base.pkl')
        return fleet

    def get_installed_fleet(self):
        if os.path.exists('./data/Installed_base.pkl'):
            fleet = pd.read_pickle('./data/Installed_base.pkl')
        else:
            fleet= self._fetch_installed_base()
        return fleet
