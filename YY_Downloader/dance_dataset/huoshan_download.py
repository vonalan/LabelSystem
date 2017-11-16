import json
from collections import OrderedDict
from os.path import isfile, join, isdir
from os import listdir, makedirs
import urllib
import time


def write_url(id, download_url_file):
    with open(download_url_file, 'a') as f3:
        f3.write(id + '\n')


def is_downloaded(id, download_url_file):
    with open(download_url_file, 'r') as f4:
        x = f4.readlines()
    tmp = id + '\n'
    if tmp in x:
        is_download = True
    else:
        is_download = False
    return is_download


def download_url(json_dir, output_dir):
    json_list = [f for f in listdir(json_dir) if isfile(join(json_dir, f))]
    for json_file in json_list:
        tmp = join(json_dir, json_file)
        with open(tmp, 'r') as f1:
            x = json.load(f1, object_pairs_hook=OrderedDict)
        level_page = join(output_dir, json_file)
        if not isdir(level_page):
            makedirs(level_page)
        data_list = x['data']
        for i in data_list:
            try:
                video_id = str(i['data']['id'])
            except Exception:
                continue
            #mv_name = i['data']['text']
            #mv_name = mv_name.replace('.', '')
            #if mv_name == '':
            mv_name = video_id
            download_mv_dir = join(level_page, mv_name)
            if is_downloaded(video_id, r'D:\Users\Administrator\Desktop\huoshan_download.txt'):
                print '%s is downloaded.' % video_id
                continue
            if not isdir(download_mv_dir):
                try:
                    makedirs(download_mv_dir)
                except Exception:
                    continue
            json_file1 = join(download_mv_dir, video_id + '.json')
            write_json = json.dumps(i)
            with open(json_file1, 'w') as f2:
                f2.write(str(write_json))
            download_mv_name = join(download_mv_dir, video_id)
            video_url = i['data']['video']['url_list'][1]
            urllib.urlretrieve(video_url, download_mv_name + '.mp4')
            pic_url = i['data']['video']['cover_medium']['url_list'][0]
            urllib.urlretrieve(pic_url, download_mv_name + '.webp')
            write_url(video_id, r'D:\Users\Administrator\Desktop\huoshan_download.txt')
            print video_id
            time.sleep(15)



if __name__ == '__main__':
    json_dir = r'D:\Users\Administrator\Desktop\huoshan'
    out_dir = r'D:\Users\Administrator\Desktop\huoshan_download'

    download_url(json_dir, out_dir)
