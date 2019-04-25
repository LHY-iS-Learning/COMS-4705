from collections import defaultdict

def build_dict(read_file):
    buildDict = defaultdict(int)
    for r in read_file:
        buildDict[r.strip().split()[0]] = int(r.strip().split()[1])
    return buildDict

class Vocab:
    def __init__(self, data_path):
        # read from files
        read_word_dict = open(data_path + '.word', 'r').read().strip().split('\n')
        read_pos_dict = open(data_path + '.pos', 'r').read().strip().split('\n')
        read_label_dict = open(data_path + '.labels', 'r').read().strip().split('\n')
        read_action_dict = open(data_path + '.actions', 'r').read().strip().split('\n')

        #build from files
        word_dict = build_dict(read_word_dict)
        pos_dict = build_dict(read_pos_dict)
        label_dict = build_dict(read_label_dict)
        action_dict = build_dict(read_action_dict)

        # assigin
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.label_dict = label_dict
        self.action_dict = action_dict


    def actionid2action_str(self, id):
        return self.actions[id]

    def action2id(self, tag):
        return self.action_dict[tag]

    def label2id(self, tag):
        return self.label_dict[tag]

    def pos2id(self, tag):
        return self.pos_dict[tag]
    
    def word2id(self, word):
        return self.word_dict[word] if word in self.word_dict else self.word_dict['<unk>']

    def num_words(self):
        return len(self.word_dict)

    def num_pos(self):
        return len(self.pos_dict)

    def num_labels(self):
        return len(self.label_dict)

    def num_actions(self):
        return len(self.action_dict)
