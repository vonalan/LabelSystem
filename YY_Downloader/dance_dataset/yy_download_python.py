import json
from collections import OrderedDict
from os.path import isfile, join, isdir, exists
from os import listdir, makedirs
import urllib
import urllib2
import math
import time
import threading

lock = threading.Lock() 

def get_video_list(filepath=''):
    with open(filepath, 'r') as rf: 
        lines = rf.readlines()
    video_list = [(item.strip().split()[0], item.strip().split()[1]) 
                    for index, item in enumerate(lines)
                    if index != 0]
    return video_list

def write_url(item, url_path):
    global lock
    lock.acquire()
    with open(url_path, 'a') as wf:
        line = '%s %s'%(item[0], item[1])
        wf.write(line)
        wf.write('\n')
    lock.release()

def download(index, item, output_dir, error_url_path):
    filename = item[1].strip().split('/')[-1]
    filepath = join(output_dir, filename)
    if not exists(filepath):
        try:
            urllib2.urlopen(item[1])
            urllib.urlretrieve(item[1], filepath)
            print "%6d -- 1 -- %s" % (index, item[1])
        except Exception:
            write_url(item, error_url_path)
            print "%6d -- 1 -- %s" % (index, item[1])

def download_them_all(video_list, output_dir, error_url_path, num_threads=1): 
    if not exists(output_dir): makedirs(output_dir)
    with open(error_url_path, 'a') as wf: 
        wf.write('resid	resurl')
        wf.write('\n')
    
    num_batches = int(math.ceil(len(video_list)/float(num_threads)))
    for i in range(num_batches): 
        start = i * num_threads
        end = min((i + 1) * num_threads, len(video_list))
        offset = end - start
        
        threads = []
        for j in range(offset): 
            item = video_list[start+j]
            th = threading.Thread(target=download, args=(start+j, item, output_dir, error_url_path))
            threads.append(th)
            th.start()
            time.sleep(1)
        for j in range(offset): 
            threads[j].join()
        
if __name__ == '__main__':
    video_list_path = 'video.txt'
    output_dir = 'videos'
    error_url_path = 'error.txt'
    
    video_list = get_video_list(video_list_path)
    download_them_all(video_list, output_dir, error_url_path, 24)
