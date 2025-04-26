from xbrl_instance import XbrlInstance
import json 
import re


class UsGaap:

    def __init__(self) -> None:
        pass


    @staticmethod
    def collect_properties(elms:list):
        """[Summary]
        :param [elms]: a list of beautifulsoup elements used to build a dictionary of the elements attributes and text values
        """
        res = {}
        for elm in elms:
            uid = f"{elm.name}|{elm.get('contextRef').split('_')[-1]}"
            res[uid] = dict(elm.attrs, **{'name':elm.name}, **{'value':elm.text})
        return res


    @staticmethod
    def map_context(soup, elms:dict):
        """Maps datetime string to an element based on it's context.period.instant. Finds the context element based on the contextRef provided in the elements properties dict.
            Period element can contain an instant element or both a start date & end date element.

        :param [res]: a dictionary of element attributes returned by collect_properties()
        """
        for k,v in elms.items():
            
            period = soup.find(name="context", attrs={"id": v.get('contextRef')}).find('period')
            
            children = period.findChildren(recursive=True)
            children_names = []
            for child in children:
                children_names.append(child.name)

            if 'instant' in children_names:
                period = period.find('instant').text
                elms[k]['period'] = period

            elif 'startDate' in  children_names and 'endDate' in children_names:
                startDate = period.find('startDate').text
                elms[k]['startdate'] = f'{startDate}'
                endDate = period.find('endDate').text
                elms[k]['enddate'] = f'{endDate}'


        return elms


    @classmethod
    def get_concepts(self, xbrli:XbrlInstance = None):
        """Returns a list of us-gaap element names contained with in the document
        """
        pattern = re.compile(r"us-gaap:.*",  re.MULTILINE)
        matches = re.findall(pattern, xbrli.response)
        res = list(set([match.split(":")[-1].replace('>','') for match in matches])) # unique values
        res.remove('explicitMember')
        res.remove('RevenueRemainingPerformanceObligationExpectedTimingOfSatisfactionStartDateAxis.domain')
        return res


    @classmethod
    def parse_gaap(self, xbrli:XbrlInstance = None):
        """Iterates through a list of concepts and constructs a dictionary for desired attributes of each concept
        """
        concepts = self.get_concepts(xbrli)

        res = dict()
        for c in concepts:
            elms = xbrli.soup.find_all(c)
            concept_properties = self.collect_properties(elms)
            concept_properties = self.map_context(xbrli.soup, concept_properties)
            res = dict(res, **concept_properties)

        parsed_res = dict()
        for concept, conceptdict in res.items():
            parsed = {key: conceptdict[key] for key in ['name', 'period', 'startdate','enddate', 'value', 'unitRef','decimals'] if key in conceptdict.keys()}
            parsed_res[concept] = parsed

        with open("./parsed_output.json", "w") as f:
            f.write(json.dumps(parsed_res, indent = 4))

        return parsed_res

    
    # TODO
    def serialize(self):
        pass