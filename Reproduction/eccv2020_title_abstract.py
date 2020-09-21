import urllib.request
import re
from pyquery import PyQuery as pq
import csv


def getHtml(url):
    page = urllib.request.urlopen(url)
    html = page.read()
    html = html.decode('utf-8')
    return html


# def download_file(download_url, file_name):
#     response = urllib.request.urlopen(download_url)
#     file = open(file_name, 'wb')
#     file.write(response.read())
#     file.close()
#     print("Completed")


save_path = 'F:/papers/eccv2020/eccv2020.csv'
# save_path = 'F:/papers/cvpr2020/'
url = 'https://www.ecva.net/papers.php'
html = getHtml(url)
# print(html)
parttern = re.compile(r'papers/eccv_2020/papers_ECCV/html/\d*_ECCV_2020_paper.php')
url_list = parttern.findall(html)
print(len(url_list))



paperTitle = []
paperAbstract = []

for i in range(484, len(url_list)):
    # name = url.split('/')[-1]
    # print(name)
    print(i)
    sub_html = getHtml('https://www.ecva.net/' + url_list[i])
    p = pq(sub_html)
    # print(p('div').filter('#papertitle').html().strip())
    paperTitle.append(p('div').filter('#papertitle').html().strip())
    paperAbstract.append(p('div').filter('#abstract').html().strip())
    addList = (p('div').filter('#papertitle').html().strip(),p('div').filter('#abstract').html().strip())
    print(addList)
    with open(save_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(addList)

print(len(paperTitle), len(paperAbstract))
paperTable = list(zip(paperTitle, paperAbstract))

# with open(save_path, 'a', newline='', encoding='utf-8') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(paperTable)
