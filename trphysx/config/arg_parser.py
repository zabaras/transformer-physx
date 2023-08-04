"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import dataclasses
from dataclasses import dataclass
import sys
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from inspect import isclass
from typing import Any, Iterable, List, Dict, Optional, Tuple, Union
from typing_extensions import Protocol


class DataClass(Protocol):
    # checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: Dict


class HfArgumentParser(ArgumentParser):
    """
    This subclass of `argparse.ArgumentParser` uses type hints on dataclasses to generate arguments.
    The class is designed to play well with the native argparse. In particular, you can add more (non-dataclass backed)
    arguments to the parser after initialization and you'll get the output back after parsing as an additional
    namespace.

    Code originally from the Huggingface Transformers repository:
    https://github.com/huggingface/transformers/blob/master/src/transformers/hf_argparser.py

    Args:
        dataclass_types (Union[DataClass, Iterable[DataClass]]):
            Dataclass type, or list of dataclass types for which 
            we will "fill" instances with the parsed args.
        kwargs (optional): Passed to `argparse.ArgumentParser()` in the regular way.
    """

    dataclass_types: Iterable[DataClass]

    def __init__(self, dataclass_types: Union[DataClass, Iterable[DataClass]], **kwargs):
        """Constructor
        """
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = dataclass_types
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)
        
        
    def _add_dataclass_arguments(self, dtype: DataClass):
        for field in dataclasses.fields(dtype):
            field_name = f"--{field.name}"
            kwargs = field.metadata.copy()
            # field.metadata is not used at all by Data Classes,
            # it is provided as a third-party extension mechanism.
            origin_type = getattr(field.type, "__origin__", field.type)
        
            if isinstance(field.type, str):
                raise ImportError(
                    "This implementation is not compatible with Postponed Evaluation of Annotations (PEP 563),"
                    "which can be opted in from Python 3.7 with `from __future__ import annotations`."
                    "We will add compatibility when Python 3.9 is released."
                )
            typestring = str(field.type)
            for prim_type in (int, float, str):
                for collection in (List,):
                    if typestring == f"typing.Union[{collection[prim_type]}, NoneType]":
                        field.type = collection[prim_type]
                if typestring == f"typing.Union[{prim_type.__name__}, NoneType]":
                    field.type = prim_type

            if isinstance(field.type, type) and issubclass(field.type, Enum):
                kwargs["choices"] = list(field.type)
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
            elif field.type is bool:
                if field.type is bool or (field.default is not None and field.default is not dataclasses.MISSING):
                    kwargs["action"] = "store_false" if field.default is True else "store_true"
                if field.default is True:
                    field_name = f"--no_{field.name}"
                    kwargs["dest"] = field.name
            # Python 3.9 fix
            elif hasattr(field.type, "__origin__") and "List" in str(field.type):
                kwargs["nargs"] = "+"
                kwargs["type"] = field.type.__args__[0]
                assert all(
                    x == kwargs["type"] for x in field.type.__args__
                ), "{} cannot be a List of mixed types".format(field.name)
                if field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
            # Python 3.10 fix
            elif isclass(origin_type) and issubclass(origin_type, list):
                kwargs["type"] = field.type.__args__[0]
                kwargs["nargs"] = "+"
                if field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
                elif field.default is dataclasses.MISSING:
                    kwargs["required"] = True
            else:
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
                else:
                    kwargs["required"] = True
            self.add_argument(field_name, **kwargs)

    def parse_args_into_dataclasses(
        self,
        args: Iterable[str] = None,
        return_remaining_strings: bool = False,
        look_for_args_file: bool = True,
        args_filename: str = None
    ) -> Tuple[DataClass]:
        """
        Parse command-line args into instances of the specified dataclass types.

        Args:
            args (Iterable[str]):
                List of strings to parse. The default is taken from sys.argv. (same as argparse.ArgumentParser)
            return_remaining_strings (bool):
                If true, also return a list of remaining argument strings.
            look_for_args_file (bool):
                If true, will look for a ".args" file with the same base name as the entry point script for this
                process, and will append its potential content to the command line args.
            args_filename (str):
                If not None, will uses this file instead of the ".args" file specified in the previous argument.
        Returns:
            (Tuple[DataClass]):
                - the dataclass instances in the same order as they were passed to the initializer.abspath
                - if applicable, an additional namespace for more (non-dataclass backed) arguments added to the parser
                  after initialization.
                - The potential list of remaining argument strings. (same as argparse.ArgumentParser.parse_known_args)
        """
        if args_filename or (look_for_args_file and len(sys.argv)):
            if args_filename:
                args_file = Path(args_filename)
            else:
                args_file = Path(sys.argv[0]).with_suffix(".args")

            if args_file.exists():
                fargs = args_file.read_text().split()
                args = fargs + \
                    args if args is not None else fargs + sys.argv[1:]
                # in case of duplicate arguments the first one has precedence
                # so we append rather than prepend.
        namespace, remaining_args = self.parse_known_args(args=args)
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype)}
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            for k in keys:
                delattr(namespace, k)
            obj = dtype(**inputs)
            outputs.append(obj)
        if len(namespace.__dict__) > 0:
            # additional namespace.
            outputs.append(namespace)
        if return_remaining_strings:
            return (*outputs, remaining_args)
        else:
            if remaining_args:
                raise ValueError(
                    f"Some specified arguments are not used by the HfArgumentParser: {remaining_args}")

            return (*outputs,)
