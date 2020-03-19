from pathlib import Path
import logging
import os
import sys
import time

import numpy as np

PATH = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULT_DIR = PATH / "../results"


class Logger(object):
    LOGGER = None

    def __init__(
        self,
        log_dir,
        unique_prefix,
        result_dir=DEFAULT_RESULT_DIR
    ):
        log_dir.mkdir(exist_ok=True)

        self.vals_dict = {}
        self.count_dict = {}
        #self.tb_handler = TensorboardHadler(log_dir, unique_prefix)
        self.csv_handler = CsvHandler(log_dir, unique_prefix)

        LOGGER = self

    @staticmethod
    def get_logger():
        return LOGGER

    def log_kv(self, key, value):
        self.vals_dict[key] = value

    def log_mean_kv(self, key, value):
        old = self.vals_dict.get(key, 0)
        cnt = self.count_dict.get(key, 0)

        self.vals_dict[key] = (old*cnt + value) / (cnt+1)
        self.count_dict[key] = cnt + 1

    def log_arr_kv(self, key, value):
        old = self.vals_dict.get(key, [])
        old.append(value)
        self.vals_dict[key] = old

    def reduce_arr_kv(self, key, new_key, func):
        val = self.vals_dict.get(key, [])
        if not (isinstance(val, list) or isinstance(val, np.ndarray)):
            raise ValueError("Cannot reduce non-array kv.")

        def safe_func(arr):
            if len(val) == 0:
                return 0
            else:
                return func(arr)

        self.log_kv(new_key, safe_func(val))

    def write_logs(self, skip_arrs=False):
        kvs = {}
        for k, v in self.vals_dict.items():
            if isinstance(v, list) or isinstance(v, np.ndarray):
                if not skip_arrs:
                    kvs[k] = v[-1]
            else:
                kvs[k] = v

        #self.tb_handler.write_logs(kvs)
        self.csv_handler.write_logs(kvs)
        write_stdout(kvs)

        self.vals_dict.clear()
        self.count_dict.clear()

    def close():
        #self.tb_handler.close()
        self.csv_handler.close()


class TensorboardHadler:
    def __init__(self, log_dir, prefix):
        path = log_dir / ('tb%s' % prefix)
        path.mkdir(exist_ok=True)
        path = path / 'events'
        self.step = 1
        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat
        self.tf = tf
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        byte_path = compat.as_bytes(str(path))
        self.writer = pywrap_tensorflow.EventsWriter(byte_path)

    def write_logs(self, kvs):
        def summary_val(k, v):
            kwargs = {'tag': k, 'simple_value': float(v)}
            return self.tf.Summary.Value(**kwargs)
        _vals = [summary_val(k, v) for k, v in kvs.items()]
        summary = self.tf.Summary(value=_vals)
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = self.step
        self.writer.WriteEvent(event)
        self.writer.Flush()
        self.step += 1

    def close(self):
        if self.writer:
            self.writer.Close()
            self.writer = None


class CsvHandler:
    def __init__(self, log_dir, prefix):
        self.path = log_dir / ('logs%s.csv' % prefix)
        self.file = self.path.open("w")
        self.header_written = False

    def write_logs(self, kvs):
        if not self.header_written:
            row = ";".join(kvs.keys()) + "\n"
            self.file.write(row)
            self.header_written = True

        row = ";".join(map(str, kvs.values())) + "\n"
        self.file.write(row)

    def close(self):
        self.file.close()


def write_stdout(kvs):
    str_dict = {}
    for k, v in kvs.items():
        if hasattr(v, "__float__"):
            valstr = "%-8.5g" % v
        else:
            valstr = str(v)

        str_dict[k] = valstr

    max_width_keys = max(map(len, str_dict.keys()))
    max_width_vals = max(map(len, str_dict.values()))
    dashes = '-' * (max_width_keys + max_width_vals + 7)
    lines = [dashes]
    for k, v in str_dict.items():
        lines.append('| %s%s | %s%s |' % (
            k,
            ' ' * (max_width_keys - len(k)),
            v,
            ' ' * (max_width_vals - len(v)),
            )
        )
    lines.append(dashes)
    sys.stdout.write('\n'.join(lines) + '\n')
    sys.stdout.flush()
