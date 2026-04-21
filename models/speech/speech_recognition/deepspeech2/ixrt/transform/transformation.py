"""Transformation module."""
import copy
import yaml
import io
import logging
import importlib
from collections import OrderedDict
from collections.abc import Sequence
from inspect import signature


import_alias = dict(
    identity="transform.functional:Identity",
    time_warp="transform.spec_augment:TimeWarp",
    time_mask="transform.spec_augment:TimeMask",
    freq_mask="transform.spec_augment:FreqMask",
    cmvn="transform.cmvn:CMVN",
    fbank="transform.spectrogram:LogMelSpectrogram",
    spectrogram="transform.spectrogram:Spectrogram",
    wav_process="transform.spectrogram:WavProcess",
    stft="transform.spectrogram:Stft",
    istft="transform.spectrogram:IStft",
    stft2fbank="transform.spectrogram:Stft2LogMelSpectrogram",
    fbank_kaldi="transform.spectrogram:LogMelSpectrogramKaldi",
    cmvn_json="transform.cmvn:GlobalCMVN")


def dynamic_import(import_path, alias=dict()):
    """dynamic import module and class

    :param str import_path: syntax 'module_name:class_name'
        e.g., 'paddlespeech.s2t.models.u2:U2Model'
    :param dict alias: shortcut for registered class
    :return: imported class
    """
    if import_path not in alias and ":" not in import_path:
        raise ValueError(
            "import_path should be one of {} or "
            'include ":", e.g. "paddlespeech.s2t.models.u2:U2Model" : '
            "{}".format(set(alias), import_path))
    if ":" not in import_path:
        import_path = alias[import_path]

    module_name, objname = import_path.split(":")
    m = importlib.import_module(module_name)
    return getattr(m, objname)


class Transformation():
    """Apply some functions to the mini-batch

    Examples:
        >>> kwargs = {"process": [{"type": "fbank",
        ...                        "n_mels": 80,
        ...                        "fs": 16000},
        ...                       {"type": "cmvn",
        ...                        "stats": "data/train/cmvn.ark",
        ...                        "norm_vars": True},
        ...                       {"type": "delta", "window": 2, "order": 2}]}
        >>> transform = Transformation(kwargs)
        >>> bs = 10
        >>> xs = [np.random.randn(100, 80).astype(np.float32)
        ...       for _ in range(bs)]
        >>> xs = transform(xs)
    """

    def __init__(self, conffile=None):
        if conffile is not None:
            if isinstance(conffile, dict):
                self.conf = copy.deepcopy(conffile)
            else:
                with io.open(conffile, encoding="utf-8") as f:
                    self.conf = yaml.safe_load(f)
                    assert isinstance(self.conf, dict), type(self.conf)
        else:
            self.conf = {"mode": "sequential", "process": []}

        self.functions = OrderedDict()
        if self.conf.get("mode", "sequential") == "sequential":
            for idx, process in enumerate(self.conf["process"]):
                assert isinstance(process, dict), type(process)
                opts = dict(process)
                process_type = opts.pop("type")
                class_obj = dynamic_import(process_type, import_alias)
                # TODO(karita): assert issubclass(class_obj, TransformInterface)
                try:
                    self.functions[idx] = class_obj(**opts)
                except TypeError:
                    try:
                        signa = signature(class_obj)
                    except ValueError:
                        # Some function, e.g. built-in function, are failed
                        pass
                    else:
                        logging.error("Expected signature: {}({})".format(
                            class_obj.__name__, signa))
                    raise
        else:
            raise NotImplementedError(
                "Not supporting mode={}".format(self.conf["mode"]))

    def __repr__(self):
        rep = "\n" + "\n".join("    {}: {}".format(k, v)
                               for k, v in self.functions.items())
        return "{}({})".format(self.__class__.__name__, rep)

    def __call__(self, xs, uttid_list=None, **kwargs):
        """Return new mini-batch

        :param Union[Sequence[np.ndarray], np.ndarray] xs:
        :param Union[Sequence[str], str] uttid_list:
        :return: batch:
        :rtype: List[np.ndarray]
        """
        if not isinstance(xs, Sequence):
            is_batch = False
            xs = [xs]
        else:
            is_batch = True

        if isinstance(uttid_list, str):
            uttid_list = [uttid_list for _ in range(len(xs))]

        if self.conf.get("mode", "sequential") == "sequential":
            for idx in range(len(self.conf["process"])):
                func = self.functions[idx]

                # TODO(karita): use TrainingTrans and UttTrans to check __call__ args
                # Derive only the args which the func has
                try:
                    param = signature(func).parameters
                except ValueError:
                    # Some function, e.g. built-in function, are failed
                    param = {}
                _kwargs = {k: v for k, v in kwargs.items() if k in param}
                try:
                    if uttid_list is not None and "uttid" in param:
                        xs = [
                            func(x, u, **_kwargs)
                            for x, u in zip(xs, uttid_list)
                        ]
                    else:
                        xs = [func(x, **_kwargs) for x in xs]

                except Exception:
                    logging.fatal("Catch a exception from {}th func: {}".format(
                        idx, func))
                    raise
        else:
            raise NotImplementedError(
                "Not supporting mode={}".format(self.conf["mode"]))

        if is_batch:
            return xs
        else:
            return xs[0]
