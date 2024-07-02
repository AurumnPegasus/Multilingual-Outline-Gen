# Where the real magic happens:
import re
import sys
import mwparserfromhell
import json

import requests

from document_type import content_type
from extract_sections import section_extraction, relevant_sections
from page_extract import page_extract, intro_extract
from preprocessing import remove_templates, cleaning
from link_extraction import ref_links
from webscrape import convert_pdf_to_txt, remove_tags

import subprocess
import xml.sax

from multiprocessing.dummy import Pool as Threadpool

import wandb
wandb.init(project='outlinetastic')

class WikiXmlHandler(xml.sax.handler.ContentHandler):
    """Content handler for Wiki XML data using SAX"""

    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._current_tag = None
        self._pages = []

    def characters(self, content):
        """Characters between opening and closing tags"""
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        """Opening tag of element"""
        if name:
            self._current_tag = name
            self._buffer = []
            self._buffer.append("<" + str(self._current_tag) + ">")

    def endElement(self, name):
        """Closing tag of element"""
        if name == self._current_tag:
            self._values[name] = ''.join(self._buffer) + "</" + str(self._current_tag) + ">"

        if name == 'page':
            self._pages.append(self._values)


# Object for handling xml
handler = WikiXmlHandler()
# Parsing object
parser = xml.sax.make_parser()
parser.setContentHandler(handler)
# Iteratively process file
no_pages = 0


def scrape_text(link_list):
    scraped_text = ""
    try:
        r = requests.get(link_list, timeout=(3, 3)).status_code  #, timeout=(5, 5)

        if int(r) == 200:
            contentt = content_type(link_list)
            # print("Link "+str(j)+" of type:",contentt)

            if contentt == 'pdf':
                scraped_text += convert_pdf_to_txt(link_list)
            elif contentt == 'html':
                scraped_text += remove_tags(link_list)
            else:
                print("Unable to scrape this url.", link_list)
                pass

            scraped_text = scraped_text.replace('\n', ' ')
            scraped_text = scraped_text.replace('\r', ' ')
            scraped_text = scraped_text.replace('\b', ' ')
            # remove excess spacings also:
            scraped_text = re.sub(r'\s\s+', ' ', scraped_text)


    except Exception as e:
        print(e)
        with open('error_urls.txt', 'a') as ert:
            ert.write(link_list + '---->' + str(e))
            ert.write('\n')
        r = 403
        pass

    return scraped_text


def pipeline(page_txt=None):
    page_txt = mwparserfromhell.parse(page_txt)

    sections, main_section_names = section_extraction(page_txt)

    # lets scrape only the relevant sections, i.e. having word count>=avg of all sections:
    relevant, relevant_indice = relevant_sections(sections)
    clean = remove_templates(relevant)

    # getting links:
    link_list = []
    for i in range(len(relevant)):
        temp_link_list = ref_links(str(relevant[i]))
        link_list.append(temp_link_list)

    output = []

    for i in range(len(link_list)):
        threadpool = Threadpool(processes=int(sys.argv[3]))

        refs = threadpool.map(scrape_text, link_list[i])
        output.append(
            {'title': main_section_names[relevant_indice[i]].strip(), 'content': clean[i], 'references': refs})

        threadpool.close()
        threadpool.join()

    return output


# pipeline() #pass xml filepath as argument here

def intro_data(page):
    intro = intro_extract(str(page))
    # intro=page  #temporary
    parsed_text = mwparserfromhell.parse(intro)

    refs = ref_links(parsed_text)
    clean_intro = cleaning(parsed_text)

    references = []
    threadpool = Threadpool(processes=int(sys.argv[3]))

    try:
        references = threadpool.map(scrape_text, refs)


    except Exception as e:
        print(e)
        references = []

    threadpool.close()
    threadpool.join()

    out = {
        "title": "Introduction",
        "content": clean_intro,
        "references": references
    }

    return out

f = open('final_titles.json', 'r')
data = json.load(f)

def main_script(xml_str,domain):
    # xml_path = 'sample_pages/sample_page.xml'

    page_txt = open(xml_str,'r').read()

    pages = page_extract(mwparserfromhell.parse(page_txt))

    # output=[]
    print(f"Total no. of pages for {domain}:", len(pages))
    # print(data.keys())
    outfile = open(f'{domain}.json', 'a')  #


    # print("Page no.:",str(i))
    # print('-----------------------')
    # print(title)
    # print('-----------------------')
    # print(title_list)
        # print('+++++++++++++++++++++++')
        # print(title)
        # print('+++++++++++++++++++++++')

    for i in range(len(pages)):
        title = re.findall("<title>(.*?)</title>", str(pages[i]), re.DOTALL)[0]
        print("Page no.:", str(i))
        intro = intro_data(pages[i])
        op = pipeline(pages[i])
        op.append(intro)
        temp = {"title": title, "sections": op}
        # output.append({"title":title,"sections":op})

        outfile.write(json.dumps(temp, ensure_ascii=False))
        outfile.write('\n')
        # flag = 1

    outfile.close()

        # if flag == 1:
        #     break

    return





def domain_list_allocate(page_str,domain_pre):
    page_txt = page_str

    pages = page_extract(mwparserfromhell.parse(page_txt))

    global data
    pre=str(domain_pre).split('/')[-1]

    for domain in list(data.keys()):
        title_list = data[domain][str(pre)]
        title_list = [' '.join(temp.split('_')) for temp in title_list]

        title = re.findall("<title>(.*?)</title>", str(pages[0]), re.DOTALL)[0]

        if title in title_list:
            with open(f'{domain_pre}_{domain}.xml', 'a') as writefile:
                writefile.write(page_str)


# iteratation:
no_pages = 0


def iterative_run(domain_pre,bz2_path=str(sys.argv[1])):
    
    for line in subprocess.Popen(['bzcat'],
                                 stdin=open(bz2_path),
                                 stdout=subprocess.PIPE).stdout:
        parser.feed(line)

        global no_pages
        # Stop when 5 articles have been found
        # flag=False

        if len(handler._pages) > no_pages:
            pagestr = ""
            no_pages += 1
            pagestr += ("<page>\n")
            for key in handler._pages[no_pages - 1]:
                # print(handler._pages[no_pages-1][key])
                pagestr += (str(handler._pages[no_pages - 1][key]) + '\n')
            pagestr += "</page>\n"

            print("Page no.", no_pages)

            # main_script(pagestr)

            domain_list_allocate(pagestr,domain_pre)

            wandb.log({'page number': no_pages})

            # print("Executed scraping script for the above pages successfully!")
            # flag = True

            # main_script(pagestr)

            # if flag:
            #     break

        # if len(handler._pages)==80:
        #     break


iterative_run(bz2_path=str(sys.argv[1]),domain_pre=str(sys.argv[2]))

#main_script(sys.argv[1],str(sys.argv[2]))
