import dynet as dynet
import random
import numpy as np

class Network:
    def __init__(self, vocab, properties):
        self.properties = properties
        self.vocab = vocab
        
        # first initialize a computation graph container (or model).
        self.model = dynet.Model()

        # assign the algorithm for backpropagation updates.
        self.updater = dynet.AdamTrainer(self.model)

        # create embeddings for words and tag features.
        self.word_embedding = self.model.add_lookup_parameters((vocab.num_words(), properties.word_embed_dim))
        self.pos_embedding = self.model.add_lookup_parameters((vocab.num_pos(), properties.pos_embed_dim))
        self.label_embedding = self.model.add_lookup_parameters((vocab.num_labels(), properties.label_embed_dim))

        # assign transfer function
        self.transfer = dynet.rectify  # can be dynet.logistic or dynet.tanh as well.

        # define the input dimension for the embedding layer.
        # here we assume to see two words after and before and current word (meaning 5 word embeddings)
        # and to see the last two predicted tags (meaning two tag embeddings)
        self.input_dim = 20 * properties.word_embed_dim + 20 * properties.pos_embed_dim + 12 * properties.label_embed_dim

        # define the hidden layer.
        self.hidden_layer1 = self.model.add_parameters((properties.hidden_dim1, self.input_dim))
        self.hidden_layer2 = self.model.add_parameters((properties.hidden_dim2, properties.hidden_dim1))

        # define the hidden layer bias term and initialize it as constant 0.2.
        self.hidden_layer1_bias = self.model.add_parameters(properties.hidden_dim1, init=dynet.ConstInitializer(0.2))
        self.hidden_layer2_bias = self.model.add_parameters(properties.hidden_dim2, init=dynet.ConstInitializer(0.2))

        # define the output weight.
        self.output_layer = self.model.add_parameters((vocab.num_actions(), properties.hidden_dim2))

        # define the bias vector and initialize it as zero.
        self.output_bias = self.model.add_parameters(vocab.num_actions(), init=dynet.ConstInitializer(0))

    def build_graph(self, features):
        # extract word and tags ids
        word_ids = [self.vocab.word2id(word) for word in features[0:20]]
        pos_ids = [self.vocab.pos2id(pos) for pos in features[20:40]]
        label_ids = [self.vocab.label2id(label) for label in features[40:]]
        
        # extract word embeddings and tag embeddings from features
        word_embeds = [self.word_embedding[wid] for wid in word_ids]
        pos_embeds = [self.pos_embedding[pid] for pid in pos_ids]
        label_embeds = [self.label_embedding[lid] for lid in label_ids]
        
        # concatenating all features (recall that '+' for lists is equivalent to appending two lists)
        embedding_layer = dynet.concatenate(word_embeds + pos_embeds + label_embeds)

        # calculating the hidden layer
        # .expr() converts a parameter to a matrix expression in dynet (its a dynet-specific syntax).
        hidden1 = self.transfer(self.hidden_layer1 * embedding_layer + self.hidden_layer1_bias)
        hidden2 = self.transfer(self.hidden_layer2 * hidden1 + self.hidden_layer2_bias)

        # calculating the output layer
        output = self.output_layer * hidden2 + self.output_bias

        # return the output as a dynet vector (expression)
        return output

    def train(self, train_file, epochs):
        # matplotlib config
        loss_values = []

        for i in range(epochs):
            print 'started epoch', (i+1)
            losses = []
            train_data = open(train_file, 'r').read().strip().split('\n')

            # shuffle the training data.
            random.shuffle(train_data)
            
            step = 0
            for line in train_data:
                fields = line.strip().split()
                features, label = fields[:-1], fields[-1]
                gold_label = self.vocab.action2id(label)
                result = self.build_graph(features)

                # getting loss with respect to negative log softmax function and the gold label.
                loss = dynet.pickneglogsoftmax(result, gold_label)

                # appending to the minibatch losses
                losses.append(loss)
                step += 1

                if len(losses) >= self.properties.minibatch_size:
                    # now we have enough loss values to get loss for minibatch
                    minibatch_loss = dynet.esum(losses) / len(losses)

                    # calling dynet to run forward computation for all minibatch items
                    minibatch_loss.forward()

                    # getting float value of the loss for current minibatch
                    minibatch_loss_value = minibatch_loss.value()

                    # printing info and plotting
                    loss_values.append(minibatch_loss_value)
                    if len(loss_values)%10==0:
                        """ ax.set_xlim([0, len(loss_values)+10])
                        ax.plot(loss_values)
                        plt.draw()
                        plt.pause(0.0001) """
                        progress = round(100 * float(step) / len(train_data), 2)
                        print 'current minibatch loss', minibatch_loss_value, 'progress:', progress, '%'

                    # calling dynet to run backpropagation
                    minibatch_loss.backward()

                    # calling dynet to change parameter values with respect to current backpropagation
                    self.updater.update()

                    # empty the loss vector
                    losses = []

                    # refresh the memory of dynet
                    dynet.renew_cg()

            # there are still some minibatch items in the memory but they are smaller than the minibatch size
            # so we ask dynet to forget them
            dynet.renew_cg()

    def decode(self, features):
        output = self.build_graph(features)
        scores = output.npvalue()
        best_action_id = np.argmax(scores)
        action = self.vocab.actionid2action_str(best_action_id)
        dynet.renew_cg()
        return action



    def load(self, filename):
        self.model.populate(filename)

    def save(self, filename):
        self.model.save(filename)
