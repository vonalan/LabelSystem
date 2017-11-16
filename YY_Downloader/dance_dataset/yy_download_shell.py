import json
from collections import OrderedDict
from os.path import isfile, join, isdir, exists
from os import listdir, makedirs
import urllib
import math
import time
import subprocess
import threading

def get_video_list(filepath=''):
    rf = open(filepath, 'r')
    video_list = [(item.strip().split()[0], item.strip().split()[1]) 
                    for index, item in enumerate(rf)
                    if index != 0]
    rf.close()
    return video_list

def gen_video_list(finepath='', video_list=[]):
    wf = open(finepath, 'w')
    for item in video_list: 
        wf.write(item[1])
        wf.write('\n')
    wf.close()

def download(url, outputDir, thread_id): 
    filename = url.strip().split('/')[-1]
    filepath = join(outputDir, filename)
    status = 0
    if not exists(filepath): 
        cmd = 'wget -O %s %s'%(filepath, url)
        status = subprocess.call(cmd, shell=True)
    return status 

def download_them_all(video_list, outputDir, num_threads=1): 
    if not exists(outputDir): makedirs(outputDir)
    num_batches = int(math.ceil(len(video_list)/float(num_threads)))
    for i in range(num_batches): 
        start = i * num_threads
        end = min((i + 1) * num_threads, len(video_list))
        offset = end - start
        
        threads = []
        for j in range(offset): 
            url = u'%s'%video_list[start+j][1]
            th = threading.Thread(target=download, args=(url, outputDir, j))
            threads.append(th)
            th.start()
            time.sleep(1)
        for j in range(offset): 
            threads[j].join()
        
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
    videoListPath = 'video.txt'
    outputDir = 'videos'
    video_list = get_video_list(videoListPath)
    download_them_all(video_list, outputDir, 24)
