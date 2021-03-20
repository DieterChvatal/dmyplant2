import arrow
import json
import base64
import requests
import logging
import os
from datetime import datetime, timedelta
import time
import pickle
import pandas as pd

maxdatapoints = 100000  # Datapoints per request, limited by Myplant


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


class MyPlant(object):

    _name = ''
    _password = ''
    _session = None
    _caching = 0

    def __init__(self, caching=7200):
        """MyPlant Constructor"""
        self._caching = caching
        # load and manage credentials from hidden file
        try:
            with open("./data/.credentials", "r", encoding='utf-8-sig') as file:
                cred = json.load(file)
            self._name = cred['name']
            self._password = cred['password']
        except FileNotFoundError:
            raise

    def deBase64(self, text):
        return base64.b64decode(text).decode('utf-8')

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
                    logging.error(f'Login {self._name} failed')
                    raise MyPlantException(
                        f'Login {self._name} failed')
            except:
                raise

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

    def asset_data(self, serialNumber):
        """
        Returns an Asset based on its id with all details
        including properties and DataItems.

        Parameters:
        Name	    type    Description
        sn          int     IB ItemNumber Engine
        ----------------------------------------------
        url: /asset?assetType=J-Engine&serialNumber=sn
        """
        return self.fetchdata(url=r"/asset?assetType=J-Engine&serialNumber=" + str(serialNumber))

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
        return self.fetchdata(url=fr"/asset/{id}/dataitem/{itemId}?timestamp={timestamp}")

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
        return self.fetchdata(url=fr"/asset/{id}/history/data?from={p_from}&to={p_to}&assetType=J-Engine&dataItemId={itemId}&timeCycle={timeCycle}&includeMinMax=false&forceDownSampling=false")

    def history_batchdata(self, id, DataItems_Request, p_from, p_to, timeCycle=3600):
        """
        url: /asset/{assetId}/dataitem/{dataItemId}
        Parameters:
        Name	    type            Description
        assetId     int64           Id of the Asset to query the DateItem for.
        itemIds     pd.DataFrame    DataItem Id's, Names & Units
        p_from      int64           timestamp start timestamp in ms.
        p_to        int64           timestamp stop timestamp in ms.
        timeCycle   int64           interval in seconds.
        """

        # prepare a komma separated string of DataItemID's
        itemIds = DataItems_Request.ID.astype(str).to_list()
        IDS = ','.join(itemIds)

        # claculate how many full rows can be downloaded per request within the limit
        rows_per_request = maxdatapoints // DataItems_Request.ID.count()

        lp_from = p_from.timestamp * 1000  # Start at p_from
        lp_to = min((lp_from + rows_per_request * timeCycle * 1000),
                    p_to.timestamp * 1000)

        df = pd.DataFrame([])
        while lp_from < p_to.timestamp * 1000:
            print(
                f"Download Data {arrow.get(lp_from).format('DD.MM.YYYY - HH:mm')} to {arrow.get(lp_to).format('DD.MM.YYYY - HH:mm')}, end: {arrow.get(p_to).format('DD.MM.YYYY - HH:mm')}", end="", flush=True)
            ldata = self.fetchdata(
                url=fr"/asset/{id}/history/batchdata?from={lp_from}&to={lp_to}&timeCycle={timeCycle}&assetType=J-Engine&includeMinMax=false&forceDownSampling=false&dataItemIds={IDS}")
            # restructure data to dict
            ds = dict()
            ds['labels'] = ['time'] + \
                DataItems_Request.myPlantName.astype(str).to_list()
            ds['data'] = [[r[0]] + [rr[0] for rr in r[1]]
                          for r in ldata['data']]

            # import to Pandas DataFrame
            ldf = pd.DataFrame(ds['data'], columns=ds['labels'])
            df = df.append(ldf)
            # calculate next cycle
            lp_from = lp_to + timeCycle * 1000
            lp_to = min((lp_to + rows_per_request *
                         timeCycle * 1000), p_to.timestamp * 1000)

            # Addtional Datetime column calculated from timestamp
            df['datetime'] = pd.to_datetime(df['time'] * 1000000)
        return df

    def gdi(self, ds, sub_key, data_item_name):
        """Unpack value from Myplant Json datastructure based on key & DataItemName"""
        if sub_key == 'nokey':
            return ds.get(data_item_name, None)
        else:
            local = {x['value']
                     for x in ds[sub_key] if x['name'] == data_item_name}
            return local.pop() if len(local) != 0 else None

    # def d(self, ts):
    #     return pd.Timestamp(ts, unit='s')

    # def future_timestamp(self, ts, hours):
    #     return int(ts + hours * 3600.0)

    # def to_myplant_ts(self, ts):
    #     return int(ts * 1000.0)

    # def from_myplant_ts(self, ts):
    #     return int(ts / 1000.0)

    @property
    def caching(self):
        """the current cache time"""
        return self._caching
