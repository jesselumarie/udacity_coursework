# -*- coding: utf-8 -*-
"""
Your task is to use the iterative parsing to process the map file and
find out not only what tags are there, but also how many, to get the
feeling on how much of which data you can expect to have in the map.
The output should be a dictionary with the tag name as the key
and number of times this tag can be encountered in the map as value.

Note that your code will be tested with a different data file than the 'example.osm'
"""
import xml.etree.ElementTree as ET
import pprint

def count_tags(filename):
    tags = {} #create dictionary
     
    
    with open(filename) as xml_file:

        tree = ET.iterparse(xml_file) #read in file 
        
        for event, node in tree: #event is an iterparse event, node is the element value

            if node.tag in tags:  #add the tag if its new, increment if its not
                tags[node.tag] += 1
            else:
                tags[node.tag] = 1

    return tags


def test():

    tags = count_tags('example.osm')
    pprint.pprint(tags)
    """assert tags == {'bounds': 1,
                     'member': 3,
                     'nd': 4,
                     'node': 20,
                     'osm': 1,
                     'relation': 1,
                     'tag': 7,
                     'way': 1} """ #took out the assertions becuase I'm now using a different dataset

    

if __name__ == "__main__":
    test()