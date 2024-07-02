# cleaning sectional text by removing certain tags and templates:

import re
import mwparserfromhell
from bs4 import BeautifulSoup


# DOESNT ALWAYS work...need to write some code to remove random symbols and signs
def strip_html_tags(text):
    """
    This function will remove all the occurrences of html tags from the text.

    arguments:
        input_text: "text" of type "String".

    return:
        value: "text" after removal of html tags.

    Example:
    Input : This is a nice place to live. <IMG>
    Output : This is a nice place to live.
    """
    # Initiating BeautifulSoup object soup.
    soup = BeautifulSoup(text, "html.parser")
    # Get all the text other than html tags.
    stripped_text = soup.get_text(separator=" ")
    return stripped_text





def remove_templates(section_text):
    filtered_section = []

    for i in range(len(section_text)):
        try:
            temp = mwparserfromhell.parse(section_text[i])

            # get list of all templates in the section which we'll remove:
            template_list = temp.filter_templates()

            # removing the unnecessary tags and the content in them (of no use from nlp standpoint):
            temp = re.sub(
                '(<format>((.|\n)*?)</format>)|(<contributor>((.|\n)*?)</contributor>)|(<timestamp>((.|\n)*?)</timestamp>)|(<ns>((.|\n)*?)</ns>)|(<id>((.|\n)*?)</id>)|(<parentid>((.|\n)*?)</parentid>)|(<ip>((.|\n)*?)</ip>)|(<comment>((.|\n)*?)</comment>|(<model>((.|\n)*?)</model>|(<sha1>((.|\n)*?)</sha1>)))',
                '', str(temp))

            temp = mwparserfromhell.parse(temp)
            # using the strip_code method to remove unnecessary attributes:
            temp = temp.strip_code().strip()

            # converting to string as operations are easier to perform:
            temp = str(temp)

            for j in template_list:
                temp = temp.replace(str(j), '')

                # remove excess \n occurences also if needed:
                temp = temp.replace('\n', ' ')

                # print("Cleaned temp:\n",temp)
                # remove excess spacings also:
                temp = re.sub(r'\s\s+', ' ',temp)
                # print("Throrughly Cleaned temp:\n", temp)

                temp=strip_html_tags(temp)
                temp = re.sub(r'\s\s+', ' ', temp)

            # Removing all html tags present using strip_html_tags() defined earlier:
            filtered_section.append(temp)

        except Exception as e:
            print(e)

    return filtered_section


def cleaning(page_text):
    temp = mwparserfromhell.parse(page_text)

    # get list of all templates in the section which we'll remove:
    template_list = temp.filter_templates()

    # removing the unnecessary tags and the content in them (of no use from nlp standpoint):
    temp = re.sub(
        '(<format>((.|\n)*?)</format>)|(<contributor>((.|\n)*?)</contributor>)|(<timestamp>((.|\n)*?)</timestamp>)|(<ns>((.|\n)*?)</ns>)|(<id>((.|\n)*?)</id>)|(<parentid>((.|\n)*?)</parentid>)|(<ip>((.|\n)*?)</ip>)|(<comment>((.|\n)*?)</comment>|(<model>((.|\n)*?)</model>|(<sha1>((.|\n)*?)</sha1>)))',
        '', str(temp))

    temp = mwparserfromhell.parse(temp)
    # using the strip_code method to remove unnecessary attributes:
    temp = temp.strip_code().strip()

    # converting to string as operations are easier to perform:
    temp = str(temp)

    for j in template_list:
        temp = temp.replace(str(j), '')

        # remove excess \n occurences also if needed:
        temp = temp.replace('\n', ' ')

        # print("Cleaned temp:\n",temp)
        # remove excess spacings also:
        temp = re.sub(r'\s\s+', ' ', temp)
        # print("Throrughly Cleaned temp:\n", temp)

        temp = strip_html_tags(temp)
        temp = re.sub(r'\s\s+', ' ', temp)

    return temp