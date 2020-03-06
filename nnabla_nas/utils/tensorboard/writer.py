import time
from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


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
            step (int, optional): Optional global step value for training process to record with the event.
            walltime: float. Optional walltime to override the default (current) walltime (from time.time()) seconds
                after epoch.
        """
        event.wall_time = time.time() if walltime is None else walltime
        if step is not None:
            event.step = int(step)
        self.event_writer.add_event(event)

    def add_summary(self, summary, global_step=None, walltime=None):
        r"""Adds a `Summary` protocol buffer to the event file.

        Args:
            summary: A `Summary` protocol buffer.
            global_step (int, optional): Optional global step value for training process to record with the summary.
            walltime (float, optional): Optional walltime to override the default (current) walltime (from time.time())
                seconds after epoch.
        """
        event = event_pb2.Event(summary=summary)
        self.add_event(event, global_step, walltime)

    def add_graph(self, graph_profile, walltime=None):
        r"""Adds a `Graph` and step stats protocol buffer to the event file.

        Args:
            graph_profile: A `Graph` and step stats protocol buffer.
            walltime (float, optional): Optional walltime to override the default (current) walltime (from time.time())
                seconds after epoch.
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
