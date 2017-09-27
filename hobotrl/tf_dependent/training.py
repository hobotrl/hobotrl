# -*- coding: utf-8 -*-

import time

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging


class RestoreVariablesHook(tf.train.SessionRunHook):

    def __init__(self,
                 checkpoint_dir, checkpoint_file=None,
                 var_list=None,
                 wait_for_checkpoint=False,
                 max_wait_secs=7200,
                 recovery_wait_secs=30,
                 ):
        super(RestoreVariablesHook, self).__init__()
        self._dir, self._file, self._var_list = checkpoint_dir, checkpoint_file, var_list
        self._wait_for_checkpoint, self._max_wait_secs, self._recovery_wait_secs = \
            wait_for_checkpoint, max_wait_secs, recovery_wait_secs
        self._saver = tf.train.Saver(var_list=self._var_list)

    def after_create_session(self, session, coord):
        super(RestoreVariablesHook, self).after_create_session(session, coord)

        if self._file:
            self._saver.restore(session, self._file)
            return
        wait_time = 0
        ckpt = tf.train.get_checkpoint_state(self._dir)
        while not ckpt or not ckpt.model_checkpoint_path:
            if self._wait_for_checkpoint and wait_time < self._max_wait_secs:
                logging.info("Waiting for checkpoint to be available.")
                time.sleep(self._recovery_wait_secs)
                wait_time += self._recovery_wait_secs
                ckpt = tf.train.get_checkpoint_state(self._dir)
            else:
                return

        # Loads the checkpoint.
        self._saver.restore(session, ckpt.model_checkpoint_path)
        self._saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)


def MonitoredTrainingSession(master='',  # pylint: disable=invalid-name
                             is_chief=True,
                             checkpoint_dir=None,
                             scaffold=None,
                             restore_var_list=None,
                             hooks=None,
                             chief_only_hooks=None,
                             save_checkpoint_secs=600,
                             save_summaries_steps=100,
                             save_summaries_secs=None,
                             config=None,
                             stop_grace_period_secs=120):
    """Creates a `MonitoredSession` for training.

    For a chief, this utility sets proper session initializer/restorer. It also
    creates hooks related to checkpoint and summary saving. For workers, this
    utility sets proper session creator which waits for the chief to
    initialize/restore.


    Args:
      master: `String` the TensorFlow master to use.
      is_chief: If `True`, it will take care of initialization and recovery the
        underlying TensorFlow session. If `False`, it will wait on a chief to
        initialize or recover the TensorFlow session.
      checkpoint_dir: A string.  Optional path to a directory where to restore
        variables.
      scaffold: A `Scaffold` used for gathering or building supportive ops. If
        not specified, a default one is created. It's used to finalize the graph.
     restore_var_list: a list of variables, optional, if not all variables should
        be recovered from checkpoint.
        Useful when changing network structures during training, i.e., finetuning
        a pretrained model with new output layers.

      hooks: Optional list of `SessionRunHook` objects.
      chief_only_hooks: list of `SessionRunHook` objects. Activate these hooks if
        `is_chief==True`, ignore otherwise.
      save_checkpoint_secs: The frequency, in seconds, that a checkpoint is saved
        using a default checkpoint saver. If `save_checkpoint_secs` is set to
        `None`, then the default checkpoint saver isn't used.
      save_summaries_steps: The frequency, in number of global steps, that the
        summaries are written to disk using a default summary saver. If both
        `save_summaries_steps` and `save_summaries_secs` are set to `None`, then
        the default summary saver isn't used.
      save_summaries_secs: The frequency, in secs, that the summaries are written
        to disk using a default summary saver.  If both `save_summaries_steps` and
        `save_summaries_secs` are set to `None`, then the default summary saver
        isn't used.
      config: an instance of `tf.ConfigProto` proto used to configure the session.
        It's the `config` argument of constructor of `tf.Session`.
      stop_grace_period_secs: Number of seconds given to threads to stop after
        `close()` has been called.

    Returns:
      A `MonitoredSession` object.
    """
    scaffold = scaffold or tf.train.Scaffold()

    if not is_chief:
        session_creator = tf.train.WorkerSessionCreator(
            scaffold=scaffold, master=master, config=config)
        return tf.train.MonitoredSession(session_creator=session_creator, hooks=hooks or [],
                                         stop_grace_period_secs=stop_grace_period_secs)

    all_hooks = []
    if chief_only_hooks:
        all_hooks.extend(chief_only_hooks)

    if restore_var_list is None:
        restore_checkpoint_dir = checkpoint_dir
    else:
        restore_checkpoint_dir = None
        all_hooks.append(RestoreVariablesHook(checkpoint_dir, var_list=restore_var_list))
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        missing_vars = filter(lambda v: not (v in restore_var_list), all_vars)
        logging.warning("MonitoredTrainingSession not restoring %s", missing_vars)
        # local_init_op = tf.group(*[v.initializer for v in missing_vars])
        # restore_scaffold = tf.train.Scaffold(local_init_op=local_init_op, saver=tf.train.Saver(restore_var_list))

    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold,
        checkpoint_dir=restore_checkpoint_dir,
        master=master,
        config=config)

    if checkpoint_dir:
        all_hooks.append(
            tf.train.StepCounterHook(output_dir=checkpoint_dir))

        if (save_summaries_steps and save_summaries_steps > 0) or (
                    save_summaries_secs and save_summaries_secs > 0):
            all_hooks.append(tf.train.SummarySaverHook(
                scaffold=scaffold,
                save_steps=save_summaries_steps,
                save_secs=save_summaries_secs,
                output_dir=checkpoint_dir))
        if save_checkpoint_secs and save_checkpoint_secs > 0:
            all_hooks.append(tf.train.CheckpointSaverHook(
                checkpoint_dir, save_secs=save_checkpoint_secs, scaffold=scaffold))

    if hooks:
        all_hooks.extend(hooks)
    return tf.train.MonitoredSession(session_creator=session_creator, hooks=all_hooks,
                                     stop_grace_period_secs=stop_grace_period_secs)
