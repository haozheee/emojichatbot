import json


class Emoji:
    unicode = ""
    name = ""
    short_code = ""
    definition = ""
    keywords = []
    categories = []
    senses = []
    '''Note: some emoji does NOT have any senses'''

    def __init__(self, unicode, name, short_code, definition, keywords, categories, senses):
        self.unicode = unicode
        self.name = name
        self.short_code = short_code
        self.definition = definition
        self.keywords = keywords
        self.categories = categories
        self.senses = senses

    @staticmethod
    def resolve_json(path):
        file = open(path, "rb")
        json_list = json.load(file)
        emoji_list = []
        index = 0
        for x in json_list:
            sense_pair = []
            adj_senses = x['senses']['adjectives']
            verb_senses = x['senses']['verbs']
            noun_senses = x['senses']['nouns']
            for group in adj_senses:
                for group_sense_key in group:
                    for sense_list in group.values():
                        for sense in sense_list:
                            sense_pair.append(('ADJ', group_sense_key, sense))
                            '''print(('ADJ', group_sense_key, sense))'''
            for group in verb_senses:
                for group_sense_key in group:
                    for sense_list in group.values():
                        for sense in sense_list:
                            sense_pair.append(('VERB', group_sense_key, sense))
                            '''print(('VERB', group_sense_key, sense))'''
            for group in noun_senses:
                for group_sense_key in group:
                    for sense_list in group.values():
                        for sense in sense_list:
                            sense_pair.append(('NOUN', group_sense_key, sense))
                            '''print(('NOUN', group_sense_key, sense))'''
            emoji_list.append(Emoji(x['unicode'], ['name'], ['shortcode',], x['definition'], x['keywords'], x['category'], sense_pair))
            print("no."+ str(index) + " emoji senses:" + str(sense_pair[:2]))
            index = index + 1
        print("Load Emoji Data Complete!")
        file.close()
        return emoji_list
