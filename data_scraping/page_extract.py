import subprocess
import xml.sax
import re
import mwparserfromhell

def page_extract(xml_input):
    query="<page>(.*?)</page>"
    pages = re.findall(query, str(xml_input), re.DOTALL)

    return pages

def intro_extract(page_text):

    try:
        query = "<text>(.*?)==[^>]+=="
        x = re.findall(query, str(page_text), re.DOTALL)[0]
        return x
    except Exception as e:
        return "NA"

# intro_extract(mwparserfromhell.parse(open('sample_pages/sample_page.xml','r').read()))
#
# print(mwparserfromhell.parse(open('sample_pages/sample_page.xml','r').read()))

