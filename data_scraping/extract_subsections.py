# modularized
# code
# better
# for reusability:



def subsection_extraction(section_text):
    subsection_dict = {}
    for i in main_sections:
        subsection_dict[i] = []

    for i in range(len(section_text)):
        # subsection headings within ===(.*?)===
        regex_for_section_names = "(?<====)(.*?)((?= \/===)|(?====))"
        tsections = list(x.group() for x in re.finditer(regex_for_section_names, section_text[i][0]))
        print("No. of subsections:", len(tsections))
        print(tsections)
        subsection_dict[main_sections[i]] = tsections

    subsections_extracted = {}

    for key, value in subsection_dict.items():
        tempd = {}
        # print(key)
        if len(value) != 0:
            for j in range(1, len(value) - 1):
                tempd[value[j - 1]] = extract_sectional_text(str(wikicode), value[j - 1], value[j])

            tempd[value[-1]] = extract_sectional_text(str(wikicode), value[-1], '')

            # print(tempd)
            subsections_extracted[key] = tempd

    print('------------------------------------------------------------------------------------------\n')
    return subsections_extracted