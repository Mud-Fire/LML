import requests
import urllib.request
import os

url_1 = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_noglobal/rois_ho/Pitt_00"
url_3 = "_rois_ho.1D"
url_2 = []
for line in open("subject_IDs.txt", "r"):
    url_2.append(line.strip())

print(url_2)

for i in range(43, len(url_2)):
    url = url_1 + url_2[i] + url_3
    print(url)
    urllib.request.urlretrieve(url)
