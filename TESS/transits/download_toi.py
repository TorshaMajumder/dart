#
# Import all the dependencies
#
import os
import requests
from splinter import Browser
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

path = os.path.dirname(os.path.abspath(__file__))

def scrape_tic_ids():

    """
    Scrape the .csv file from - "https://exofop.ipac.caltech.edu/tess/view_toi.php"

    Returns
    -------
    toi: CSV file
        Store the toi.csv file in the current folder
    """

    executable_path = {'executable_path': ChromeDriverManager().install()}
    browser = Browser('chrome', **executable_path, headless=True)

    session = requests.Session()
    base_url="https://exofop.ipac.caltech.edu/tess/"
    url = base_url+"view_toi.php"
    browser.visit(url)

    # Convert the browser.html into a BeautifulSoup object
    html = browser.html
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.body.find_all('table')

    # Download all the URLs
    download_urls=list()
    links = tables[1].find_all('a')
    
    for l in links:
        download_urls.append(l['href'])
    
    csv_url = [i for i in download_urls if "csv" in i]    
    csv_url= base_url+csv_url[0]
    # print(csv_url)

    # Download the TOIs as .csv
    req = requests.get(csv_url)
    url_content = req.content
    csv_file = open(path+'/toi.csv', 'wb')

    csv_file.write(url_content)
    csv_file.close()

    session.close()


if __name__ == '__main__':

    scrape_tic_ids()

