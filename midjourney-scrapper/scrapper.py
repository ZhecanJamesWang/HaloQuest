import requests, json, os
from datetime import datetime
from bs4 import BeautifulSoup
import pdb

def get_image_urls(url):

    image_dict = {}
    headers = {'User-Agent': 'curl/7.84.0'}
    page = requests.get(url, headers=headers, allow_redirects=True)
    soup = BeautifulSoup(page.content, 'html.parser')
    body = soup.find("body")
    sc = list(body.find_all("script"))[-1].string
    sc = str(sc)
    # parse JSON
    sc = json.loads(sc).get("props").get("pageProps").get("jobs")
    for i in sc:
        url = i.get("event").get("seedImageURL")
        prompt = i['prompt']
        image_dict[i['id']] = [url, prompt]
    return image_dict

def download_image(image_dict):
    curdate = datetime.now().strftime("%Y%m%d")
    os.makedirs(curdate, exist_ok=True)
    os.chdir(curdate)

    headers = {'User-Agent': 'curl/7.84.0'}
    i = 1
    for id in image_dict:
        try:
            [url, prompt] = image_dict[id]
            with open(id + '.txt', 'w') as f:
                if str(prompt) == 'nan':
                    prompt = ''
                f.write(prompt)
            page = requests.get(url, headers=headers, allow_redirects=True)
            with open(id + ".png", 'wb') as f:
                f.write(page.content)
            i+=1
        except Exception as e:
            print(e)

        print(f"Downloaded {i}")

url1 = 'https://www.midjourney.com/showcase/top'
url2 = 'https://www.midjourney.com/app/feed/?sort=rising'
# url3 = 'https://www.midjourney.com/showcase/recents'

image_dict1 = get_image_urls(url1)
image_dict2 = get_image_urls(url2)
# image_dict3 = get_image_urls(url3)
image_dict1.update(image_dict2)
# image_dict1.update(image_dict3)

download_image(image_dict1)
