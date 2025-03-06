# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

import numpy as np
import src.envs.encoders as encoders
import src.envs.generators as generators


from torch.utils.data import DataLoader
from src.dataset import EnvDataset

from ..utils import bool_flag


SPECIAL_WORDS = ["<eos>", "<pad>", "<sep>", "(", ")"]
SPECIAL_WORDS = SPECIAL_WORDS + [f"<SPECIAL_{i}>" for i in range(10)]

logger = getLogger()


class InvalidPrefixExpression(Exception):
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)

def data_type_to_encoder(params, typ):
    tensor_dim = typ.count('[')
    assert tensor_dim == typ.count(']')
    if typ.startswith('int'):
        ext = typ[3:]
        if tensor_dim > 0:
            dims = [int(elt.strip('[')) for elt in ext.strip(']').split(']')]
            max_dim = max(dims)
            assert ext == ''.join(f'[{d}]' for d in dims)
            return encoders.NumberArray(params, max_dim, 'V', tensor_dim)
        else:
            assert typ == 'int'
            return encoders.PositionalInts(params.base)
    elif typ.startswith('range'):
        assert tensor_dim == 0, "at the moment we don't support arrays of ranges, use int arrays"
        if ',' in typ:
            low, high = map(int, typ[6:-1].split(','))
            assert typ == f'range({low},{high})'
        else:
            low = 0
            high = int(typ[6:-1])
            assert typ == f'range({high})'
        return encoders.SymbolicInts(low, high  - 1) # SymbolicInts is inclusive on the high end
    else:
        assert False, "type not supported"






class ArithmeticEnvironment(object):

    TRAINING_TASKS = {"arithmetic"}

    def __init__(self, params):
        self.max_len = params.max_len
        self.operation = params.operation
        
        self.base = params.base
        self.max_class = params.max_class

        #if self.operation == 'data':
            #assert params.data_types, "argument --data_types is required"
            #i, o = params.data_types.split(':')
            #self.input_encoder = data_type_to_encoder(params, i)
            #self.output_encoder = data_type_to_encoder(params, o)
        #    self.input_encoder = encoders.NumberArray(params, 5, 'V', 1)
        #    self.output_encoder = encoders.SymbolicInts(0, 10)
        #    dims=[]
        #    self.generator = generators.Sequence(params, dims)
        self.export_pred = params.export_pred
        self.n_eval_metrics = params.n_eval_metrics
        self.n_error_metrics = params.n_error_metrics


        if self.operation == 'matrix_rank':
            dims = [params.dim1, params.dim2]
            max_dim = 100
            tensor_dim = 2
            self.output_encoder = encoders.SymbolicInts(1, max_dim)
        else:
            dims = []
            max_dim =  4 if self.operation in ["fraction_compare", "fraction_determinant", "fraction_add", "fraction_product"] else 2
            tensor_dim =  1
            if self.operation in  ["fraction_add", "fraction_product", "fraction_simplify"]:
                self.output_encoder = encoders.NumberArray(params, 2, 'V', tensor_dim )
            elif self.operation in ["fraction_round", "gcd", "fraction_determinant","modular_add","modular_mul","elliptic"]:
                self.output_encoder = encoders.PositionalInts(params.base)
            else:
                self.output_encoder = encoders.SymbolicInts(0, 1)
        self.input_encoder = encoders.NumberArray(params, max_dim, 'V', tensor_dim)
        assert not self.export_pred or isinstance(self.output_encoder, (encoders.SymbolicInts, encoders.PositionalInts))

        self.generator = generators.Sequence(params, dims)

        # vocabulary
        self.words = SPECIAL_WORDS + sorted(list(
            set(self.input_encoder.symbols+self.output_encoder.symbols)
        ))
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        self.sep_index = params.sep_index = 2
        
        logger.info(f"words: {self.word2id}")

    def input_to_infix(self, lst):
        return ' '.join(lst)
        
    def output_to_infix(self, lst):
        return ' '.join(lst)
        
    def gen_expr(self, data_type=None):
        """
        Generate pairs of problems and solutions.
        Encode this as a prefix sentence
        """
        gen = self.generator.generate(self.rng, data_type)
        if gen is None:
            return None
        x_data, y_data = gen
        # encode input
        x = self.input_encoder.encode(x_data)
        # encode output
        y = self.output_encoder.encode(y_data)
        if self.max_len > 0 and (len(x) >= self.max_len or len(y) >= self.max_len):
            return None
        return x, y

    def decode_class(self, i):
        """
        The code class splits the test data in to subgroups by code_class
        """
        if i>=1000:
            return str(i//1000)+"-"+str(i%1000)
        return str(i)

    def code_class(self, xi, yi):
        """
        The code class splits the test data in to subgroups by code_class
        This is passed to the evaluator, so it needs to be an integer
        """
        if self.export_pred:
            v = self.output_encoder.decode(yi)
            assert v is not None
            if v >= self.max_class:
                v = self.max_class
            return v

        if self.operation in ["fraction_add", "fraction_product", "fraction_simplify", "fraction_round", "fraction_determinant"]:
            return 0
        elif self.operation in ["gcd", "modular_add", "modular_mul"]:
            v = self.output_encoder.decode(yi)
            assert v is not None
            if v >= self.max_class:
                v = self.max_class
            return v
        else:
            v = self.output_encoder.decode(yi)
            assert v is not None
            if isinstance(self.output_encoder, encoders.NumberArray):
                v = v[0]
            return v % self.max_class

    def check_prediction(self, src, tgt, hyp):
        w = self.output_encoder.decode(hyp)
        if w is None:
            return -1,[],[], None if self.export_pred else -1,[],[]
        if len(hyp) == 0 or len(tgt) == 0:
            return -1,[],[], None if self.export_pred else -1,[],[]
        if hyp == tgt:
            return 2,[],[], w if self.export_pred else 2,[],[]

        a, b, c = self.generator.evaluate(self.input_encoder.decode(src), self.input_encoder.decode(tgt), w)
        return a, b, c, w if self.export_pred else a, b, c


    def create_train_iterator(self, task, data_path, params):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=True,
            params=params,
            path=data_path,
            type = "train",
        )
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 1800),
            batch_size=params.batch_size,
            num_workers=(
                params.num_workers
                if data_path is None or params.num_workers == 0
                else 1
            ),
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    def create_test_iterator(
        self, data_type, task, data_path, batch_size, params, size
    ):
        """
        Create a dataset for this environment.
        """
        #assert data_type in ["valid", "test"] or data_type[:4] == "test"
        logger.info(f"Creating {data_type} iterator for {task} ...")
        if data_path is None:
            path_iter = None
        elif data_type == "valid":
            path_iter = data_path[0]
        elif data_type == "test":
            path_iter = data_path[1]
        else: 
            path_iter = data_path[int(data_type[4:])]
        dataset = EnvDataset(
            self,
            task,
            train=False,
            params=params,
            path=path_iter,
            size=size,
            type=data_type,
        )
        return DataLoader(
            dataset,
            timeout=0,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument(
            "--operation", type=str, default="data", help="Operation to perform"
        )
        parser.add_argument(
            "--data_types", type=str, default="", help="Data type for input and out output separated by :, e.g. \"int[5]:range(2)\""
        )
        parser.add_argument(
            "--dim1", type=int, default=10, help="Lines of matrix"
        )
        parser.add_argument(
            "--dim2", type=int, default=10, help="Columns of matrix"
        )

        
        parser.add_argument(
            "--maxint", type=int, default=1000000, help="Maximum value of integers"
        )
        parser.add_argument(
            "--minint", type=int, default=1, help="Minimum value of integers (uniform generation only)"
        )
        


        parser.add_argument(
            "--two_classes", type=bool_flag, default=False, help="Two classes in train set"
        )
        parser.add_argument(
            "--first_class_size", type=int, default=1000000, help="Standard deviation, in examples"
        )
        parser.add_argument(
            "--first_class_prob", type=float, default=0.25, help="Proportion of repeated fixed examples in train set"
        )

        parser.add_argument(
            "--base", type=int, default=1000, help="Encoding base"
        )
        parser.add_argument(
            "--modulus", type=int, default=67, help="Modulus for modular operations"
        )

        parser.add_argument(
            "--n_eval_metrics", type=int, default=0, help="number of eval metrics, returned by generator.evaluate()")

        parser.add_argument(
            "--n_error_metrics", type=int, default=0, help="number of error metrics, returned by generator.evaluate()")

        parser.add_argument(
            "--export_pred", type=bool_flag, default=False, help="export model predictions, returned by check_predictions()")
      
        parser.add_argument(
            "--max_class", type=int, default=101, help="Maximum class for reporting with error predictions"
        )

