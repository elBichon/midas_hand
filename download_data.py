from selenium import webdriver
#from selenium.webdriver.firefox.options import Options
import datetime
import bs4 as bs
from bs4 import BeautifulSoup
import urllib.request
from random import randint
from time import sleep
from selenium.webdriver.support.select import Select

now = datetime.datetime.now()
start_date = str(now.day)+'/'+str(now.month)+'/'+str(now.year-2)
epa_list = []

browser = webdriver.Firefox()
browser.get('https://www.abcbourse.com/marches/aaz.aspx')
bet_fa = browser.find_element_by_id("dlIndices")
bet_fa.send_keys('xcac40p')
html = browser.page_source
soup = BeautifulSoup(html, "html.parser")
for data in soup.findAll('td',{'class':'srd'}):
    start = '<td class="srd"><a href="/'
    end = '">'
    try:
        print((str(data).split(start))[1].split(end)[0])
        epa_list.append((str(data).split(start))[1].split(end)[0])
    except:
        continue
print(epa_list)

i = 0
while i < 3:#len(epa_list):
    browser.get('https://www.abcbourse.com/download/download.aspx?s='+str(epa_list[i].replace('cotation/','')))
    browser.implicitly_wait(10)
    sleep_time = 3 + randint(5,10)
    sleep(sleep_time)
    bet_fa = browser.find_element_by_id("txtFrom")
    bet_fa.clear()
    bet_fa.send_keys(start_date)
    select_fr = Select(browser.find_element_by_id("dlFormat"))
    select_fr.select_by_index(4)
    submit_button = browser.find_elements_by_id('Button1')[0]
    submit_button.click()
    i += 1
