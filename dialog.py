import re
import nltk


class Dialog:
    max_dialog_len = 0
    all_tokens = []
    all_text = []  # used for obtaining frequency distribution
    word_id_dict = {}
    lemon = None    # it is the lemmatizer
    read_len = 700  # -1 to read until ends
    freq_dist = None
    vocab_size = 800
    contractions = {
        "ain't": "am not / are not",
        "aren't": "are not / am not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "gon'na": "going to",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had / he would",
        "he'd've": "he would have",
        "he'll": "he shall / he will",
        "he'll've": "he shall have / he will have",
        "he's": "he has / he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how has / how is",
        "i'd": "I had / I would",
        "i'd've": "I would have",
        "i'll": "I shall / I will",
        "i'll've": "I shall have / I will have",
        "i'm": "I am",
        "i've": "I have",
        "isn't": "is not",
        "it'd": "it had / it would",
        "it'd've": "it would have",
        "it'll": "it shall / it will",
        "it'll've": "it shall have / it will have",
        "it's": "it has / it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she had / she would",
        "she'd've": "she would have",
        "she'll": "she shall / she will",
        "she'll've": "she shall have / she will have",
        "she's": "she has / she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as / so is",
        "that'd": "that would / that had",
        "that'd've": "that would have",
        "that's": "that has / that is",
        "there'd": "there had / there would",
        "there'd've": "there would have",
        "there's": "there has / there is",
        "they'd": "they had / they would",
        "they'd've": "they would have",
        "they'll": "they shall / they will",
        "they'll've": "they shall have / they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had / we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what shall / what will",
        "what'll've": "what shall have / what will have",
        "what're": "what are",
        "what's": "what has / what is",
        "what've": "what have",
        "when's": "when has / when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where has / where is",
        "where've": "where have",
        "who'll": "who shall / who will",
        "who'll've": "who shall have / who will have",
        "who's": "who has / who is",
        "who've": "who have",
        "why's": "why has / why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had / you would",
        "you'd've": "you would have",
        "you'll": "you shall / you will",
        "you'll've": "you shall have / you will have",
        "you're": "you are",
        "you've": "you have",
    }

    def __init__(self, character1, character2, text):
        self.character1 = character1
        self.character2 = character2
        self.text = text

    @staticmethod
    def pre_text(text):
        for word in text.split():
            if word.lower() in Dialog.contractions:
                text = text.replace(word, Dialog.contractions[word.lower()])
        result_tokens = []
        for token in nltk.word_tokenize(text):
            if token.isnumeric():
                result_tokens.append('1.01')    # replace any numeric token with 1.01 to reduce vocab size
            else:
                result_tokens.append(Dialog.lemon.lemmatize(token))
        return result_tokens

    @staticmethod
    def resolve_data(conversation_path, lines_path, vocab_size=800):
        dialog_list = []
        line_pairs = {}
        Dialog.lemon = nltk.WordNetLemmatizer()
        Dialog.vocab_size = vocab_size
        with open(lines_path, encoding="utf8", errors='ignore', mode="r") as lines_file:
            for count, line in enumerate(lines_file):
                parts = line.split(' +++$+++ ')
                string = str(parts[-1])
                string = string.lower()
                line_pairs[parts[0]] = string
        with open(conversation_path, "r") as conversation_file:
            for count, line in enumerate(conversation_file):
                if count > Dialog.read_len != -1:
                    break
                parts = line.split(' +++$+++ ')
                text_ids = re.findall(r'(L[0-9]+)', parts[3])
                texts = []
                for text_id in text_ids:
                    text_string = line_pairs[text_id]
                    text_tokens = Dialog.pre_text(text_string)
                    texts.append(text_tokens)
                    if len(text_tokens) > Dialog.max_dialog_len:
                        Dialog.max_dialog_len = len(text_tokens)
                    for token in text_tokens:
                        Dialog.all_text.append(token)
                dialog_list.append(Dialog(parts[0], parts[1], texts))
                if count % 100 == 0:
                    print("Load text resources, line NO." + str(count))
        print("Movie Conversation Loading Complete!")
        Dialog.max_dialog_len = Dialog.max_dialog_len + 3
        return dialog_list

    @staticmethod
    def load_word2ids():
        Dialog.freq_dist = nltk.FreqDist(Dialog.all_text)
        Dialog.all_tokens = list(dict(Dialog.freq_dist.most_common(Dialog.vocab_size + 2)).keys())
        known = sum(list(dict(Dialog.freq_dist.most_common(Dialog.vocab_size + 2)).values()))
        total = sum(list(Dialog.freq_dist.values()))
        known_rate = float(known) / float(total)        # to measure how many words are Not replaced by UNKNOWN
        print("Vocabulary Known Rate is: " + str(known_rate))
        Dialog.word_id_dict = {token: index + 3 for index, token in set(enumerate(Dialog.all_tokens))}
        Dialog.word_id_dict['TSTSTARTTST'] = 1
        Dialog.word_id_dict['TSTUNKNOWNTST'] = 2
        '''leave 0 as the empty token for padding'''

    @staticmethod
    def pair_qa_ids(dialog_list):
        dialog_x = []
        dialog_y = []
        for dialog in dialog_list:
            for index in range(len(dialog.text)):
                if index + 1 < len(dialog.text):
                    text1 = dialog.text[index]
                    text1ids = Dialog.tokens2id(text1)
                    text2 = dialog.text[index + 1]
                    text2ids = Dialog.tokens2id(text2)
                    dialog_x.append(text1ids)
                    dialog_y.append(text2ids)
        return dialog_x, dialog_y

    @staticmethod
    def pair_qa(dialog_list):
        dialog_x = []
        dialog_y = []
        for dialog in dialog_list:
            for index in range(len(dialog.text)):
                if index + 1 < len(dialog.text):
                    text1tokens = [token for token in dialog.text[index]]
                    pad_template = [0] * Dialog.max_dialog_len
                    pad_template[:len(text1tokens)] = text1tokens
                    text2tokens = [token for token in dialog.text[index + 1]]
                    dialog_x.append(text1tokens)
                    dialog_y.append(text2tokens)
        return dialog_x, dialog_y

    @staticmethod
    def id2word(id):
        if id < 0.000001:
            return ''
        for k, v in Dialog.word_id_dict.items():  # for key, value in dictionary.iteritems()
            if id == v:
                return k
        return 'ERROR'

    @staticmethod
    def word2id(word):
        if len(word) == 0:
            return 0
        if word in Dialog.word_id_dict:
            return Dialog.word_id_dict[word]
        else:
            return Dialog.word_id_dict['TSTUNKNOWNTST']

    @staticmethod
    def sent2id(sent):
        id_sent = []
        sent = Dialog.pre_text(sent)
        for word in sent:
            id_sent.append(Dialog.word2id(word))
        pad = [0] * Dialog.max_dialog_len
        pad[:len(id_sent)] = id_sent
        return pad

    @staticmethod
    def id2sent(sent):
        word_sent = []
        for i in sent:
            if i == 0:
                continue
            word_sent.append(Dialog.id2word(i))
        return ' '.join(word_sent)

    @staticmethod
    def tokens2id(sent):
        id_sent = []
        for word in sent:
            id_sent.append(Dialog.word2id(word))
        pad = [0] * Dialog.max_dialog_len
        pad[:len(id_sent)] = id_sent
        return pad



