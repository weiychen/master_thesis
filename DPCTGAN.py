import os
import warnings

from ctgan import CTGANSynthesizer
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.ctgan import Discriminator, Generator, optim
import numpy as np
import pandas as pd
import torch
from tqdm import trange
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from opacus.layers import dp_lstm
import numpy as np
# from tensorflow.keras.layers import Dense, LSTM, Embedding
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
import opacus
from tqdm import trange, tqdm
torch.manual_seed(1)

import logging
rootLogger = logging.getLogger()


# log = xes_importer.apply('ETM_Configuration2.xes')#('financial_log.xes')
# dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
# def infer_time(dataframe):
#     return -dataframe['time:timestamp'].diff(-1).dt.total_seconds()
# duration= dataframe.groupby('case:concept:name').apply(infer_time)
# dataframe['duration'] =duration.droplevel(0)

"""
reference: 
- https://opacus.ai/api/dp_lstm.html#
- https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
"""

class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 20
        self._embedding_dim = 20
        self.num_layers = 3

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self._embedding_dim,
        )
#         self.lstm = nn.LSTM(
#             input_size=self.lstm_size,
#             hidden_size=self.lstm_size,
#             num_layers=self.num_layers,
#             dropout=0.2,
#         )
        self.lstm = dp_lstm.DPLSTM(
            self.lstm_size ,#embedding size,
            self.lstm_size,#hidden size
            num_layers=self.num_layers,
            bidirectional=False,
            batch_first=True,
        )
        self.fc = dp_lstm.LSTMLinear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        
#         args,
    ):
#         self.args = args
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()
        self.sequence_length = 10#50
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        self._dataset = (
            # 'datasets/ETM_Configuration2.xes',
            'datasets/financial_log.xes',
        )[0]
        log = xes_importer.apply(self._dataset)
        dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
        series = dataframe.groupby('case:concept:name')['concept:name'].apply(list)
        test = []
        for s in series.values:
            s.insert(0,'start')
            s.append('end')
            test.append(s)
        tokens = np.concatenate(test).ravel()
#         print(tokens)
        return tokens

    def get_uniq_words(self):
        word_counts = Counter(self.words)
#         print(word_counts)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+self.sequence_length+1]),
        )
"""
duplicate reading
"""
# log = xes_importer.apply('ETM_Configuration2.xes')#('financial_log.xes') #('ETM_Configuration2.xes')
# dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)



class MyDataSampler(DataSampler):

    def generate_cond_from_condition_column_info(self, batch, data, org_data, epochs):
        """ Generate fake activities sequence (inlcuding trace). 
        E.g. A, B, D1,... """
        self.dataset = Dataset()
        self.epochs = epochs

        model = self.get_fitted_model(batch, data, org_data, epochs)

        groups = org_data.groupby(['case:concept:name']).count()
        min_constraint = min(groups['time:timestamp'])
        max_constraint = max(groups['time:timestamp'])

        text = 'start'
        next_words = len(data)*10
        model.eval()

        words = text.split(' ')
        state_h, state_c = model.init_state(len(words))

        rootLogger.info("Generating words.")
        for i in trange(next_words):
            x = torch.tensor([[self.dataset.word_to_index[w] for w in words[i:]]])
            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            words.append(self.dataset.index_to_word[word_index])

        rootLogger.info("Formatting words into dataframe.")
        ids = list()
        _words = list()
        list_index = 0
        for word in tqdm(words):
            if word == "start":
                new_ids = list()
                new_words = list()
                store_end = False
                
                continue
            if word == "end":      
                if store_end == True:
                    continue
                list_index += 1
                ids += new_ids
                _words += new_words
                store_end = True
                continue
            new_ids.append(list_index)
            new_words.append(word)
        df = pd.DataFrame({"id": ids, "word": _words})
        df.id = df.id.astype(int)
        id_groups = df.groupby(['id']).count()
        matched_id = id_groups.index[(id_groups.word >= min_constraint) & (id_groups.word <= max_constraint)]
        cleaned_df = pd.DataFrame()
        unique_traces = len(org_data['case:concept:name'].unique())
        if len(matched_id) > unique_traces:
            matched_id = matched_id[0:unique_traces]
        for id in matched_id:
            cleaned_df=cleaned_df.append(df[df.id == id])  
        cleaned_df = cleaned_df.rename(columns={"word": "concept:name","id":"traces"})    
        activities_pd = pd.get_dummies(cleaned_df['concept:name'], prefix='Activity')
        vec = activities_pd.to_numpy()
        # unique_activities = list(activities_pd.columns) 
        # vec = activities_pd[:batch].to_numpy()
        # data = data[batch:]
        return vec, cleaned_df 

    def get_fitted_model(self, batch, data, org_data, epochs):
        from checkpoint import Checkpoint, LSTMSaveLoad
        dataset_name = os.path.basename(self.dataset._dataset).split(".")[0]
        checkpoint = Checkpoint("nn_models", LSTMSaveLoad(), "")
        checkpoint.add_info("dataset", dataset_name)
        checkpoint.add_info("epochs", epochs)

        RETRAIN = False
        
        """ Load an already fitted model from file or fit a new one. """
        if checkpoint.exists() and not RETRAIN:
            rootLogger.info("Loading trained nn.Model from '{}'".format(checkpoint.save_file))
            return checkpoint.load()
        else:
            rootLogger.info("Retraining model...")
            model = self._fit_model(batch, data, org_data, epochs)
            checkpoint.save(model)
            return model

    def _fit_model(self, batch, data, org_data, epochs):
        model = Model(self.dataset)
        model.train()
        batch_size = 10#50
        sequence_length = 10#50
        max_epochs = epochs

        # Privacy engine hyper-parameters
        max_per_sample_grad_norm = 1.0
        # delta = 0#8e-5
        epsilon = 1.0
        secure_rng = False
        sample_rate = batch_size / len(data)

        dataloader = DataLoader(self.dataset, batch_size=batch_size,drop_last = True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.008)


        privacy_engine = PrivacyEngine(
            model,
            sample_rate=sample_rate,
            max_grad_norm=max_per_sample_grad_norm,
        #     target_delta=delta,
            target_epsilon=epsilon,
            epochs=epochs,
        #     secure_rng=secure_rng,
        )
        privacy_engine.attach(optimizer)

        rootLogger.info("Sampling activities")
        epochs_info = list()
        for epoch in trange(max_epochs):
            state_h, state_c = model.init_state(sequence_length)

            for _batch, (x, y) in enumerate(dataloader):
                optimizer.zero_grad()

                y_pred, (state_h, state_c) = model(x, (state_h, state_c))
                loss = criterion(y_pred.transpose(1, 2), y)

                state_h = state_h.detach()
                state_c = state_c.detach()

                loss.backward()
                optimizer.step()

            epochs_info.append({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
        
        for info in epochs_info:
            rootLogger.info(info)

        return model
        

class DPCTGAN(CTGANSynthesizer):

    def fit(self, train_data, org_data, discrete_columns=tuple(), epochs=None, disabled_dp=False):
        """
        Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # opacus parameters
        self.sigma = 5#sigma
        self.disabled_dp = disabled_dp
        self.target_delta = None#target_delta
        self.max_per_sample_grad_norm = 1#max_per_sample_grad_norm
        self.epsilon = 2 #epsilon
        self.epsilon_list = []
        self.alpha_list = []
        self.loss_d_list = []
        self.loss_g_list = []
        self.verbose = True #verbose
        self.loss = "cross_entropy" #loss

        # if self.loss != "cross_entropy":
        #     # Monkeypatches the _create_or_extend_grad_sample function when calling opacus
        #     opacus.supported_layers_grad_samplers._create_or_extend_grad_sample = ( _custom_create_or_extend_grad_sample)
        self.org_data = org_data.fillna(0)

        self._validate_discrete_columns(train_data, discrete_columns)
        self.data = train_data

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        self._data_sampler = MyDataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        if not self.disabled_dp:
            privacy_engine = opacus.PrivacyEngine( 
                discriminator,
                batch_size=self._batch_size,
                sample_rate = self._batch_size / len(self.data),
                # sample_size=train_data.shape[0],
                # alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                # noise_multiplier=self.sigma,
                max_grad_norm=self.max_per_sample_grad_norm,
                target_epsilon=self.epsilon,
                epochs = epochs
                # clip_per_layer=True,
            )
            privacy_engine.attach(optimizerD)

        # real_label = 1
        # fake_label = 0
        # criterion = nn.BCELoss()
        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1
        steps_per_epoch = max(len(train_data) // self._batch_size, 1)

        for i in trange(epochs):
            for id_ in range(steps_per_epoch):
                """missing n"""
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:                      
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    if self.disabled_dp:
                        pen = discriminator.calc_gradient_penalty(
                            real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad()
                    if self.disabled_dp:
                        pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy
                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

            if self.verbose:
                rootLogger.info(f"Epoch {i+1}, Loss G: {loss_g.detach().cpu(): .4f},"
                                f"Loss D: {loss_d.detach().cpu(): .4f}")

    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.
        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.
        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        # if condition_column is not None and condition_value is not None:
        #     condition_info = self._transformer.convert_column_name_value_to_id(
        #         condition_column, condition_value)
        #     global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
        #         condition_info, self._batch_size, self.data)
                
        # else:
        #     global_condition_vec = None
        rootLogger.info("Generating activities...")
        EPOCHS_LSTM = 100
        global_condition_vec, activities = self._data_sampler.generate_cond_from_condition_column_info(
                 self._batch_size, self.data, self.org_data, EPOCHS_LSTM)
        activities_copy = activities.copy()


        rootLogger.info("Generating durations...")
        rootLogger.info(f"Device is: {self.device}")

        MAX_TRIES = 200
        failed = False
        
        fake_batch_size = 4 # self._batch_size
        
        steps = n // fake_batch_size + 1
        data = []
        for i in trange(steps):

            next_activities = activities_copy[:fake_batch_size]
            activities_copy = activities_copy[fake_batch_size:]

            # Iterate until the generated activities match next_activities
            for j in range(MAX_TRIES):

                mean = torch.zeros(fake_batch_size, self._embedding_dim)
                std = mean + 1
                fakez = torch.normal(mean=mean, std=std).to(self._device)

                if global_condition_vec is not None:
                    condvec = global_condition_vec[:fake_batch_size]
                    if len(condvec) != fake_batch_size:
                        fakez = fakez[:len(condvec)]

                c1 = condvec # One-Hot Encoding Vector
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                generated = self._transformer.inverse_transform(fakeact.detach().cpu().numpy())
                activities_match = generated['concept:name'].values == next_activities['concept:name'].values
                activities_match = activities_match if isinstance(activities_match, bool) else activities_match.all()
                last_try = j == MAX_TRIES-1
                if activities_match or last_try:
                    # The generated activities match. Continue with next step
                    data.append(generated)

                    # Remove used entries of global_condition_vec for next iteration step
                    global_condition_vec = global_condition_vec[fake_batch_size:]

                    if not last_try or activities_match:
                        # rootLogger.info(f"Found activites match after {j+1} tries.")
                        break
                    else:
                        rootLogger.info(f"\nCouldn't find matching activities vector after {MAX_TRIES} tries...\n"
                            "Decrease the batch size or increase number of epochs and try again.")
                        failed = True
                
            if failed:
                # break
                # Just continue to see how bad the result actually is.
                pass

        data = np.concatenate(data, axis=0)
        transformed = pd.DataFrame(data[:n], columns=["concept:name", "duration"])
        
        return transformed, activities

