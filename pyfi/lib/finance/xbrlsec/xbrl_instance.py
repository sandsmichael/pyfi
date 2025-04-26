import sys
import requests
from bs4 import BeautifulSoup
import re

class XbrlInstance():

    def __init__(self, url:str=None, fp:str=None):
        """A bs4 object representing an xml document
        
        :param [url]: web link to xml verion of sec edgar financial report: i.e. https://www.sec.gov/Archives/edgar/data/1018724/000101872422000023/amzn-20220930_htm.xml
            Data will be downloaded and stored to local file @ ./output.xml

        :param [fp]: local path to xml file

        """     
           
        self.fp = fp
        self.url = url

        if self.url is not None and self.fp is not None:
            return ValueError('[Error] - Please provide a url <or> a file path; not both.')

        self.soup = self.get()
        self.ns = self.get_namespaces()
        self.accounting_standard = self.get_accounting_standard()

        # print(self.ns)
        # print(self.accounting_standard)


    def get(self):
        """ returns BeautifulSoup object representing the parsed xml document
        """
        if self.url is not None:
            header = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest"
            }
            name = self.url.split('/')[-1]
            self.response = requests.get(self.url, headers=header).text
            with open(f'./xml/{name}', 'w') as file:
                file.write(self.response)

            with open(f'./xml/{name}', 'r') as file:
                self.response = file.read()

        elif self.url is None and self.fp is not None:
            with open(self.fp, 'r') as file:
                self.response = file.read()            

        soup = BeautifulSoup(self.response, features='xml')
        return soup


    def get_namespaces(self):
        """ Returns namespace dictionary of xml document
        """        
        elm = self.soup.find('xbrl')
        ns = elm.attrs   # r"xmlns:(.*?)=")
        return ns


    def get_accounting_standard(self):
        """Determine if xml document is using us-gaap or ifrs-full accounting standards
        """
        if 'xmlns:us-gaap' in self.ns.keys():
            return 'us-gaap'
        elif 'xmlns:ifrs-full' in self.ns.keys():
            return 'ifrs-full'
        else:
            return None



