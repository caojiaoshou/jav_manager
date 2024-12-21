import hashlib
import pickle
import time
from uuid import uuid4

import httpx

with open('../result.pickle', 'rb') as f:
    result = pickle.load(f)

for segment in result['segments']:
    print(segment['no_speech_prob'])
    print(segment['text'])
    if segment['no_speech_prob'] <= 0.55:
        while True:
            text = segment['text']
            salt = uuid4().hex
            appid = '20241218002231705'
            key = 'QMXTG9JvmiLklqGUu223'

            concat_sign = appid + text + salt + key

            sign = hashlib.md5(concat_sign.encode('utf-8')).hexdigest()
            query = {
                'q': text,
                'from': 'jp',
                'to': 'zh',
                'appid': appid,
                'salt': salt,
                'sign': sign
            }
            result_2 = httpx.get('https://fanyi-api.baidu.com/api/trans/vip/translate', params=query)
            print(result_2.status_code)
            result_3 = result_2.json()
            print(result_3)
            try:
                segment['translation'] = result_2.json()['trans_result'][0]['dst']
                break
            except KeyError:
                ...
            finally:
                time.sleep(0.2)
    else:
        segment['translation'] = segment['text']

with open('../translation.pickle', 'wb') as f:
    pickle.dump(result, f)
