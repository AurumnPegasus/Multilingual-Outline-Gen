import urllib
from urllib.request import Request, urlopen
import wikipedia
import signal
import json
import requests
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from tqdm import tqdm
from bs4 import BeautifulSoup
# from PyPDF2 import PdfFileReader
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter,resolve1
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import PyPDF2

from io import StringIO
import io

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def handler(signum, frame):
    print("Forever is over!")
    return TimeoutError

signal.signal(signal.SIGALRM, handler)
def remove_tags(url):
    html_page = requests.get(url,timeout=3)
    soup = BeautifulSoup(html_page.content, "html.parser")
    text = []
    # for data in soup(['style', 'script']):
    for data in soup.find_all('p'):
        text.append(data.get_text())

    for data in soup.find_all('article'):
        text.append(data.get_text())

    for data in soup.find_all('span'):
        text.append(data.get_text())


    return ' '.join(text)[:10000]

def return_read_webpage(req):
    signal.alarm(5)
    try:
        return urlopen(req, timeout=2).read()
    except:
        signal.alarm(0)
        return TimeoutError


def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)

    # with open(path, 'rb') as fp:
    # fp=open(path, 'rb')
    # print("For path:",path)

    req = Request(
        path,
        headers={'User-Agent': 'Mozilla/5.0'},unverifiable=True)
    try:
        # webpage = return_read_webpage(req)
        webpage = urlopen(req, timeout=3).read()
        # f = urllib.request.urlopen(path).read()
        # print("using fp = io.BytesIO(webpage):")
        # signal.alarm(0)
        fp = io.BytesIO(webpage)

        read_pdf = PyPDF2.PdfFileReader(fp)
        number_of_pages = read_pdf.getNumPages()
        # print("number of pages:",number_of_pages)
        if number_of_pages>20:
            # print("Skipping this pdf as pages nos. >20")
            return ""

        # text=""
        #
        # for i in range(number_of_pages):
        #     page = read_pdf.getPage(i)
        #     page_content = page.extractText()
        #     text+=page_content+" "

        parser = PDFParser(fp)
        document = PDFDocument(parser)

        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        caching = True
        pagenos = set()


        for page in PDFPage.get_pages(fp, pagenos, password=password,caching=caching, check_extractable=True):
            # print("reading page in pdf:")

            interpreter.process_page(page)


        # print("getting text from retstr.getvalue():")
        text = retstr.getvalue()

    except TimeoutError:
        # print("time limit exceeded so skipping the link")
        device.close()
        retstr.close()
        # signal.alarm(0)
        return ""

    except Exception as e:
        device.close()
        retstr.close()
        # signal.alarm(0)
        print('here:',e)
        return ""

    device.close()
    retstr.close()
    return text[:10000]

# testing by passing a url:

#1) for pdf text:
# print('output',convert_pdf_to_txt('https://web.archive.org/web/20170525141614/http://nclm.nic.in/shared/linkimages/NCLM52ndReport.pdf'))  #http://www.africau.edu/images/default/sample.pdf
# https://fsi.nic.in/isfr2017/uttar-pradesh-isfr-2017.pdf
#https://web.archive.org/web/20170525141614/http://nclm.nic.in/shared/linkimages/NCLM52ndReport.pdf
#2) for html text:
# print(remove_tags('https://web.archive.org/web/20190215050340/https://hindi.timesnownews.com/world/article/hindi-to-become-third-language-used-in-abu-dhabi-dubai-court-system/363296'))