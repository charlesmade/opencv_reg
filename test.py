# -*- coding: UTF-8 -*-
import time
import urllib
import json
import hashlib
import base64


def main():
    f = open("../../img/1.jpg", 'rb')
    file_content = f.read()
    base64_image = base64.b64encode(file_content)
    body = bytes(urllib.parse.urlencode({'image': base64_image}), encoding='utf8')

    url = 'http://webapi.xfyun.cn/v1/service/v1/ocr/handwriting'
    api_key = '68dc7bc42b8cc2477573dac5510d33c7'
    param = {"language": "en", "location": "true"}

    x_appid = '5ba9c41e'
    str_param = bytes(json.dumps(param).replace(' ', ''), encoding='utf8')
    x_param = str(base64.b64encode(str_param), encoding='utf-8')

    x_time = int(int(round(time.time() * 1000)) / 1000)
    md5_bytes = bytes(api_key + str(x_time) + x_param, encoding='utf8')
    x_checksum = hashlib.md5(md5_bytes).hexdigest()

    x_header = {'X-Appid': x_appid,
                'X-CurTime': x_time,
                'X-Param': x_param,
                'X-CheckSum': x_checksum}
    print(x_header)
    print(type(x_header))
    return
    req = urllib.request.Request(url, body, x_header)
    response = urllib.request.urlopen(req)
    print(response.read())
    return


if __name__ == '__main__':
    main()