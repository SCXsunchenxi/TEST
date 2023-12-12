import torch
import numpy


class ContrastiveLoss(torch.nn.modules.loss._Loss):
    """
    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    """

    def __init__(self, compared_length, nb_random_samples, negative_penalty, text_prototype=False
                 ):
        super(ContrastiveLoss, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty
        self.text_prototype = text_prototype

    def forward(self, batch, encoder, train, save_memory=False):

        batch_size = batch.size(0)
        train_size = train.size(0)
        length = min(self.compared_length, train.size(2))

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        samples = numpy.random.choice(
            train_size, size=(self.nb_random_samples, batch_size)
        )
        samples = torch.LongTensor(samples)

        # Choice of length of positive and negative samples
        length_pos_neg = numpy.random.randint(low=1, high=length + 1)

        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = numpy.random.randint(
            low=length_pos_neg, high=length + 1
        )  # Length of anchors
        beginning_batches = numpy.random.randint(
            low=0, high=length - random_length + 1, size=batch_size
        )  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        beginning_samples_pos = numpy.random.randint(
            low=0, high=random_length - length_pos_neg + 1, size=batch_size
        )  # Start of positive samples in the anchors
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + length_pos_neg

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = numpy.random.randint(
            low=0, high=length - length_pos_neg + 1,
            size=(self.nb_random_samples, batch_size)
        )

        representation = encoder(torch.cat(
            [batch[
             j: j + 1, :,
             beginning_batches[j]: beginning_batches[j] + random_length
             ] for j in range(batch_size)]
        ))  # Anchors representations

        positive_representation = encoder(torch.cat(
            [batch[
             j: j + 1, :, end_positive[j] - length_pos_neg: end_positive[j]
             ] for j in range(batch_size)]
        ))  # Positive samples representations

        size_representation = representation.size(1)
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        if self.text_prototype == False:
            loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                representation.view(batch_size, 1, size_representation),
                positive_representation.view(batch_size, size_representation, 1)
            )))
        else:
            save_memory = False
            loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                representation.view(batch_size, 1, size_representation),
                positive_representation.view(batch_size, size_representation, 1)
            )) + torch.nn.functional.logsigmoid(torch.bmm(
                representation.view(batch_size, 1, size_representation),
                self.text_prototype.view(batch_size, size_representation, 1)))
                               )

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            negative_representation = encoder(
                torch.cat([train[samples[i, j]: samples[i, j] + 1][
                           :, :,
                           beginning_samples_neg[i, j]:
                           beginning_samples_neg[i, j] + length_pos_neg
                           ] for j in range(batch_size)])
            )
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    negative_representation.view(
                        batch_size, size_representation, 1
                    )
                ))
            )
            # If required, backward through the first computed term of the loss
            # and free from the graph everything related to the negative sample
            # Leaves the last backward pass to the training procedure
            if save_memory and i != self.nb_random_samples - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()

        # Text aligned loss

        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        # loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
        #     representation.view(batch_size, 1, size_representation),
        #     positive_representation.view(batch_size, size_representation, 1)
        # )))

        return loss


class ContrastiveLossVaryingLength(torch.nn.modules.loss._Loss):
    """
    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    """

    def __init__(self, compared_length, nb_random_samples, negative_penalty, text_prototype=False):
        super(ContrastiveLossVaryingLength, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty
        self.text_prototype = text_prototype

    def forward(self, batch, encoder, train, save_memory=False):
        batch_size = batch.size(0)
        train_size = train.size(0)
        max_length = train.size(2)

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        samples = numpy.random.choice(
            train_size, size=(self.nb_random_samples, batch_size)
        )
        samples = torch.LongTensor(samples)

        # Computation of the lengths of the relevant time series
        with torch.no_grad():
            lengths_batch = max_length - torch.sum(
                torch.isnan(batch[:, 0]), 1
            ).data.cpu().numpy()
            lengths_samples = numpy.empty(
                (self.nb_random_samples, batch_size), dtype=int
            )
            for i in range(self.nb_random_samples):
                lengths_samples[i] = max_length - torch.sum(
                    torch.isnan(train[samples[i], 0]), 1
                ).data.cpu().numpy()

        # Choice of lengths of positive and negative samples
        lengths_pos = numpy.empty(batch_size, dtype=int)
        lengths_neg = numpy.empty(
            (self.nb_random_samples, batch_size), dtype=int
        )
        for j in range(batch_size):
            lengths_pos[j] = numpy.random.randint(
                1, high=min(self.compared_length, lengths_batch[j]) + 1
            )
            for i in range(self.nb_random_samples):
                lengths_neg[i, j] = numpy.random.randint(
                    1,
                    high=min(self.compared_length, lengths_samples[i, j]) + 1
                )

        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = numpy.array([numpy.random.randint(
            lengths_pos[j],
            high=min(self.compared_length, lengths_batch[j]) + 1
        ) for j in range(batch_size)])  # Length of anchors
        beginning_batches = numpy.array([numpy.random.randint(
            0, high=lengths_batch[j] - random_length[j] + 1
        ) for j in range(batch_size)])  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        # Start of positive samples in the anchors
        beginning_samples_pos = numpy.array([numpy.random.randint(
            0, high=random_length[j] - lengths_pos[j] + 1
        ) for j in range(batch_size)])
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + lengths_pos

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = numpy.array([[numpy.random.randint(
            0, high=lengths_samples[i, j] - lengths_neg[i, j] + 1
        ) for j in range(batch_size)] for i in range(self.nb_random_samples)])

        representation = torch.cat([encoder(
            batch[
            j: j + 1, :,
            beginning_batches[j]: beginning_batches[j] + random_length[j]
            ]
        ) for j in range(batch_size)])  # Anchors representations

        positive_representation = torch.cat([encoder(
            batch[
            j: j + 1, :,
            end_positive[j] - lengths_pos[j]: end_positive[j]
            ]
        ) for j in range(batch_size)])  # Positive samples representations

        size_representation = representation.size(1)
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            representation.view(batch_size, 1, size_representation),
            positive_representation.view(batch_size, size_representation, 1)
        )))

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            negative_representation = torch.cat([encoder(
                train[samples[i, j]: samples[i, j] + 1][
                :, :,
                beginning_samples_neg[i, j]:
                beginning_samples_neg[i, j] + lengths_neg[i, j]
                ]
            ) for j in range(batch_size)])
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    negative_representation.view(
                        batch_size, size_representation, 1
                    )
                ))
            )
            # If required, backward through the first computed term of the loss
            # and free from the graph everything related to the negative sample
            # Leaves the last backward pass to the training procedure
            if save_memory and i != self.nb_random_samples - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()

        return loss

