import torch
import os
from itertools import cycle


class Corpus(object):
    def __init__(self, path, prefix='', ext='.txt') -> None:
        self.train = os.path.join(path, f'{prefix}train{ext}')
        self.valid = os.path.join(path, f'{prefix}valid{ext}')
        self.test = os.path.join(path, f'{prefix}test{ext}')

        self._train_num_lines = len(list(self._get_raw_iter(self.train)))
        self._test_num_lines = len(list(self._get_raw_iter(self.test)))
        if os.path.exists(self.valid):
            self._valid_num_lines = len(list(self._get_raw_iter(self.valid)))
    
    def _get_raw_iter(self, path):
        with open(path, 'r') as f:
            for line in f:
                if len(line.strip()) > 0: # Skip empty lines
                    yield line 
                else:
                    continue
    
    def get_data(self):
        # train_iter = RawTextIterableDataset(self._train_num_lines, self._get_raw_iter(self.train))
        # valid_iter = RawTextIterableDataset(self._valid_num_lines, self._get_raw_iter(self.valid))
        # test_iter = RawTextIterableDataset(self._test_num_lines, self._get_raw_iter(self.test))

        # train_iter = MyTextIterableDataset(self.train)
        # valid_iter = MyTextIterableDataset(self.valid)
        # test_iter = MyTextIterableDataset(self.test)

        train_data = MyTextDataset(self.train)
        test_data = MyTextDataset(self.test)
        if os.path.exists(self.valid):
            valid_data = MyTextDataset(self.valid)
        else:
            valid_data = None

        return train_data, valid_data, test_data
    
    def print_info(self):
        train_data, valid_data, test_data = self.get_data()
        print(f'Number of lines in train: {len(train_data)}')
        if valid_data:
            print(f'Number of lines in valid: {len(valid_data)}')
        print(f'Number of lines in test: {len(test_data)}')


class MyTextIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_path):
        """Initiate the dataset abstraction.
        """
        super(MyTextIterableDataset, self).__init__()
        self.file_path = file_path
        self.num_lines = len(list(self.parse_file(self.file_path)))
        self.current_pos = None
        self._iterator = self.parse_file(self.file_path)
    
    def parse_file(self, path):
        with open(path, 'r') as f:
            for line in f:
                if len(line.strip()) > 0: # Skip empty lines
                    yield line 
    
    def get_stream(self, path):
        return cycle(self.parse_file(path))

    def __iter__(self):
        # return self.get_stream(self.file_path)
        return self
    
    def __next__(self):
        if self.current_pos == self.num_lines - 1:
            self._iterator = self.parse_file(self.file_path)
            self.current_pos = None # go back to first line
            raise StopIteration
        item = next(self._iterator)
        if self.current_pos is None:
            self.current_pos = 0
        else:
            self.current_pos += 1
        return item

    def __len__(self):
        return self.num_lines

    def pos(self):
        """
        Returns current position of the iterator. This returns None
        if the iterator hasn't been used yet.
        """
        return self.current_pos


class MyTextDataset(): # A map-style dataset (non-iterable)
    def __init__(self, file_path):
        """Initiate the dataset abstraction.
        """
        super(MyTextDataset, self).__init__()
        self.file_path = file_path
        self.all_lines = self.read_lines(self.file_path)
        self.num_lines = len(self.all_lines)
    
    def read_lines(self, path):
        lines = []
        with open(path, 'r') as f:
            for line in f:
                if len(line.strip()) > 0: # Skip empty lines
                    lines.append(line)
        return lines
    
    def __len__(self):
        return self.num_lines

    def __getitem__(self, index):
        return self.all_lines[index]
        

class RawTextIterableDataset(torch.utils.data.IterableDataset):
    """Defines an abstraction for raw text iterable datasets.
    Adopted from https://github.com/pytorch/text/blob/master/torchtext/data/datasets_utils.py
        slightly changed
    """
    def __init__(self, full_num_lines, iterator):
        """Initiate the dataset abstraction.
        """
        super(RawTextIterableDataset, self).__init__()
        self.full_num_lines = full_num_lines
        self._iterator = iterator
        self.num_lines = full_num_lines
        self.current_pos = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_pos == self.num_lines - 1:
            raise StopIteration
        item = next(self._iterator)
        if self.current_pos is None:
            self.current_pos = 0
        else:
            self.current_pos += 1
        return item

    def __len__(self):
        return self.num_lines

    def pos(self):
        """
        Returns current position of the iterator. This returns None
        if the iterator hasn't been used yet.
        """
        return self.current_pos