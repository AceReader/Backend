import xml.etree.ElementTree as ET
import json
import nltk  # for sentence parser


def parseXML(in_path, out_path, filename):
    """Parse .xml file generated by grobid.

    Args:
        in_path (string): Input path of .xml file.
        out_path (string): Output path.
        filename (string): Name of input file.

    Returns:
        dic: Content of input file in json format.
    """

    filename = filename[:-4]+'.tei.xml'

    tree = ET.parse(in_path+filename)
    root = tree.getroot()

    # xmlns is prefix of tag
    # Use xpath for namespace
    ns = {'tei_xns': 'http://www.tei-c.org/ns/1.0'}

    dic = {}

    for child in root.find('.//tei_xns:teiHeader/tei_xns:fileDesc/tei_xns:titleStmt', ns):
        title = child.text
        title = title.lower().split()
        for i, w in enumerate(title):
            title[i] = w[0].upper()+w[1:]
        title = ' '.join(title)
        dic['title'] = title

    dic['authors'] = []
    dic['emails'] = []
    dic['orgs'] = []
    dic['country'] = []
    for analytics in root.find('.//tei_xns:teiHeader/tei_xns:fileDesc/tei_xns:sourceDesc/tei_xns:biblStruct/tei_xns:analytic', ns):
        # print(child.tag)
        for analytic in analytics:
            if 'persName' in analytic.tag:
                name_temp = []
                for names in analytic:
                    name_temp.append(names.text)
                name_temp = ' '.join(name_temp)
                dic['authors'].append(name_temp)

            if 'email' in analytic.tag:
                dic['emails'].append(analytic.text)

            if 'affiliation' in analytic.tag:
                org_temp = []
                for addresses in analytic:
                    if 'orgName' in addresses.tag:
                        org_temp.append(addresses.text)

                    if 'address' in addresses.tag:
                        for address in addresses:
                            if 'country' in address.tag:
                                dic['country'].append(address.text)

                org_temp = ', '.join(org_temp)
                dic['orgs'].append(org_temp)

    # for child in root.iterfind('.//tei_xns:title', ns):
    #    print(child.text)

    dic['keywords'] = []
    for profileDescs in root.find('.//tei_xns:teiHeader/tei_xns:profileDesc', ns):
        if 'textClass' in profileDescs.tag:
            for keywords in profileDescs.iterfind('.//tei_xns:term', ns):
                dic['keywords'].append(keywords.text)
                # print(keywords.text)

        if 'abstract' in profileDescs.tag:
            for p in profileDescs.iterfind('.//tei_xns:p', ns):
                dic['abstract'] = nltk.sent_tokenize(p.text)
                # print(p.text)

    # table fails
    dic['content'] = []
    for body in root.find('.//tei_xns:text/tei_xns:body', ns):
        for div in body:
            dic['content'].append(div.text)

    with open(out_path+filename[:-8]+'.json', 'w') as f:
        json.dump(dic, f)

    return dic
