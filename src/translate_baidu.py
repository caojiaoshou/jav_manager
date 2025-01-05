import hashlib
import time
import typing
from uuid import uuid4

import httpx


def translate_list(list_to_translate: typing.Iterable[str]) -> list[str]:
    result_list = []
    for item in list_to_translate:
        if item:
            while True:
                text = item
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
                    result_list.append(result_2.json()['trans_result'][0]['dst'])
                    break
                except KeyError:
                    ...
                finally:
                    time.sleep(0.2)
        else:
            result_list.append(item)

    return result_list
