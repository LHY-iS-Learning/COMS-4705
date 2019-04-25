from optparse import OptionParser
from network import *
import pickle
from net_properties import *
from utils import *
from vocab import *

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="train_file", metavar="FILE", default='data/vocabs')
    parser.add_option("--train_data", dest="train_data_file", metavar="FILE", default='data/train.data')
    parser.add_option("--test", dest="test_file", metavar="FILE", default=None)
    parser.add_option("--model", dest="model_path", metavar="FILE", default=None)
    parser.add_option("--vocab", dest="vocab_path", metavar="FILE", default='outputs/vocab')
    parser.add_option("--output", dest="output_file", metavar="FILE", default=None)

    parser.add_option("--we", type="int", dest="we", default=64)
    parser.add_option("--pe", type="int", dest="pe", default=32)
    parser.add_option("--le", type="int", dest="le", default=32)
    parser.add_option("--hidden1", type="int", dest="hidden1", default=200)
    parser.add_option("--hidden2", type="int", dest="hidden2", default=200)
    parser.add_option("--minibatch", type="int", dest="minibatch", default=1000)

    parser.add_option("--epochs", type="int", dest="epochs", default=7)

    (options, args) = parser.parse_args()

    if options.train_file and options.train_data_file and options.model_path and options.vocab_path:
        net_properties = NetProperties(options.we, options.pe, options.le, options.hidden1, options.hidden2, options.minibatch)

        # creating vocabulary file
        vocab = Vocab(options.train_file)

        # writing properties and vocabulary file into pickle
        pickle.dump((vocab, net_properties), open(options.vocab_path, 'w'))

        # constructing network
        network = Network(vocab, net_properties)

        # training
        network.train(options.train_data_file,options.epochs)

        # saving network
        network.save(options.model_path)
