# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.event_pb2 import Event, SessionLog
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.step_stats_pb2 import DeviceStepStats, StepStats
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.proto.versions_pb2 import VersionDef
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from .nnabla_graph import GraphVisitor


class FileWriter(object):

    r"""Write protocol buffers to event files.

    Args:
        log_dir (str): Directory where event file will be written.
        max_queue (int, optional): Size of the queue for pending events and summaries before one of the 'add' calls
            forces a flush to disk. Defaults to 10.
        flush_secs (int, optional): How often, in seconds, to flush the pending events and summaries to disk. Defaults
            to every two minutes (120s).
        filename_suffix (str, optional): Suffix added to all event filenames in the log_dir directory.
    """

    def __init__(self, log_dir, max_queue=10, flush_secs=120, filename_suffix=''):
        log_dir = str(log_dir)
        self.event_writer = EventFileWriter(
            log_dir, max_queue, flush_secs, filename_suffix)

    def get_logdir(self):
        r"""Returns the directory where event file will be written."""
        return self.event_writer.get_logdir()

    def add_event(self, event, step=None, walltime=None):
        r"""Adds an event to the event file.

        Args:
            event: An `Event` protocol buffer.
            step (int, optional): Optional global step value for training process to record with the
                event.
            walltime: float. Optional walltime to override the default (current) walltime
                (from time.time()) seconds after epoch.
        """
        event.wall_time = time.time() if walltime is None else walltime
        if step is not None:
            event.step = int(step)
        self.event_writer.add_event(event)

    def add_summary(self, summary, global_step=None, walltime=None):
        r"""Adds a `Summary` protocol buffer to the event file.

        Args:
            summary: A `Summary` protocol buffer.
            global_step (int, optional): Optional global step value for training process to record
                with the summary.
            walltime (float, optional): Optional walltime to override the default (current) walltime
                (from time.time()) seconds after epoch.
        """
        event = event_pb2.Event(summary=summary)
        self.add_event(event, global_step, walltime)

    def add_graph(self, graph_profile, walltime=None):
        r"""Adds a `Graph` and step stats protocol buffer to the event file.

        Args:
            graph_profile: A `Graph` and step stats protocol buffer.
            walltime (float, optional): Optional walltime to override the default (current) walltime
                (from time.time()) seconds after epoch.
        """
        graph = graph_profile[0]
        stepstats = graph_profile[1]
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self.add_event(event, None, walltime)

        trm = event_pb2.TaggedRunMetadata(
            tag='step1', run_metadata=stepstats.SerializeToString())

        event = event_pb2.Event(tagged_run_metadata=trm)
        self.add_event(event, None, walltime)

    def flush(self):
        r"""Flushes the event file to disk."""
        self.event_writer.flush()

    def close(self):
        r"""Flushes the event file to disk and close the file."""
        self.event_writer.close()

    def reopen(self):
        r"""Reopens the EventFileWriter."""
        self.event_writer.reopen()


class SummaryWriter(object):
    r"""Creates a `SummaryWriter` that will write out events and summaries to the event file.

    Args:
        log_dir (string): Save directory location. Default is
            runs/CURRENT_DATETIME_HOSTNAME, which changes after each run.
            Use hierarchical folder structure to compare
            between runs easily. e.g. pass in 'runs/exp1', 'runs/exp2', etc.
            for each new experiment to compare across them.
        comment (string): Comment log_dir suffix appended to the default `log_dir`. If `log_dir` is assigned, this
            argument has no effect.
        purge_step (int): Note that crashed and resumed experiments should have the same ``log_dir``.
        max_queue (int): Size of the queue for pending events and
            summaries before one of the 'add' calls forces a flush to disk.
            Default is ten items.
        flush_secs (int): How often, in seconds, to flush the
            pending events and summaries to disk. Default is every two minutes.
        filename_suffix (string): Suffix added to all event filenames in
            the log_dir directory. More details on filename construction in
            tensorboard.summary.writer.event_file_writer.EventFileWriter.
    """

    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix=''):
        if not log_dir:
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join(
                'runs', current_time + '_' + socket.gethostname() + comment)
        self.log_dir = log_dir
        self.purge_step = purge_step
        self.max_queue = max_queue
        self.flush_secs = flush_secs
        self.filename_suffix = filename_suffix

        # Initialize the file writers, but they can be cleared out on close
        # and recreated later as needed.
        self.file_writer = self.all_writers = None
        self._get_file_writer()

    def _get_file_writer(self):
        """Returns the default FileWriter instance. Recreates it if closed."""
        if self.all_writers is None or self.file_writer is None:
            self.file_writer = FileWriter(self.log_dir, self.max_queue,
                                          self.flush_secs, self.filename_suffix)
            self.all_writers = {self.file_writer.get_logdir(): self.file_writer}
            if self.purge_step is not None:
                most_recent_step = self.purge_step
                self.file_writer.add_event(
                    Event(step=most_recent_step, file_version='brain.Event:2'))
                self.file_writer.add_event(
                    Event(step=most_recent_step, session_log=SessionLog(status=SessionLog.START)))
                self.purge_step = None
        return self.file_writer

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        r"""Add a scalar value."""
        scalar_value = float(scalar_value)
        self._get_file_writer().add_summary(
            Summary(value=[Summary.Value(tag=tag, simple_value=scalar_value)]),
            global_step, walltime
        )

    def add_image(self, tag, img, global_step=None, walltime=None):
        r"""Add an image."""
        self._get_file_writer().add_summary(
            Summary(value=[Summary.Value(tag=tag, image=img)])
        )

    def add_graph(self, model, *args, **kargs):
        visitor = GraphVisitor(model, *args, **kargs)
        stepstats = RunMetadata(step_stats=StepStats(dev_stats=[DeviceStepStats(device="/device:CPU:0")]))
        graph = GraphDef(node=visitor._graph, versions=VersionDef(producer=22))
        self._get_file_writer().add_graph((graph, stepstats))

    def flush(self):
        """Flushes the event file to disk.
        Call this method to make sure that all pending events have been written to
        disk.
        """
        if self.all_writers is None:
            return
        for writer in self.all_writers.values():
            writer.flush()

    def close(self):
        if self.all_writers is None:
            return  # ignore double close
        for writer in self.all_writers.values():
            writer.flush()
            writer.close()
        self.file_writer = self.all_writers = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
