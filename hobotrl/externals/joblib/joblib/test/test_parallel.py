"""
Test the parallel module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2010-2011 Gael Varoquaux
# License: BSD Style, 3 clauses.

import os
import sys
import time
import mmap
import threading
from math import sqrt
from time import sleep
from multiprocessing import TimeoutError

import joblib
from joblib import dump, load
from joblib import parallel

from joblib.test.common import np, with_numpy
from joblib.test.common import with_multiprocessing
from joblib.testing import (parametrize, raises, check_subprocess_call,
                            SkipTest, warns)
from joblib._compat import PY3_OR_LATER

try:
    import cPickle as pickle
    PickleError = TypeError
except ImportError:
    import pickle
    PickleError = pickle.PicklingError


if PY3_OR_LATER:
    PickleError = pickle.PicklingError

try:
    from queue import Queue
except ImportError:
    # Backward compat
    from Queue import Queue

try:
    import posix
except ImportError:
    posix = None

from joblib._parallel_backends import SequentialBackend
from joblib._parallel_backends import ThreadingBackend
from joblib._parallel_backends import MultiprocessingBackend
from joblib._parallel_backends import LokyBackend
from joblib._parallel_backends import SafeFunction
from joblib._parallel_backends import WorkerInterrupt

from joblib.parallel import Parallel, delayed
from joblib.parallel import register_parallel_backend, parallel_backend
from joblib.parallel import effective_n_jobs, cpu_count

from joblib.parallel import mp, BACKENDS, DEFAULT_BACKEND
from joblib.my_exceptions import JoblibException


ALL_VALID_BACKENDS = [None] + sorted(BACKENDS.keys())
# Add instances of backend classes deriving from ParallelBackendBase
ALL_VALID_BACKENDS += [BACKENDS[backend_str]() for backend_str in BACKENDS]
PROCESS_BACKENDS = ['multiprocessing', 'loky']
PARALLEL_BACKENDS = PROCESS_BACKENDS + ['threading']

if hasattr(mp, 'get_context'):
    # Custom multiprocessing context in Python 3.4+
    ALL_VALID_BACKENDS.append(mp.get_context('spawn'))

DefaultBackend = BACKENDS[DEFAULT_BACKEND]


def get_workers(backend):
    return getattr(backend, '_pool', getattr(backend, '_workers', None))


def division(x, y):
    return x / y


def square(x):
    return x ** 2


class MyExceptionWithFinickyInit(Exception):
    """An exception class with non trivial __init__
    """
    def __init__(self, a, b, c, d):
        pass


def exception_raiser(x, custom_exception=False):
    if x == 7:
        raise (MyExceptionWithFinickyInit('a', 'b', 'c', 'd')
               if custom_exception else ValueError)
    return x


def interrupt_raiser(x):
    time.sleep(.05)
    raise KeyboardInterrupt


def f(x, y=0, z=0):
    """ A module-level function so that it can be spawn with
    multiprocessing.
    """
    return x ** 2 + y + z


def _active_backend_type():
    return type(parallel.get_active_backend()[0])


def parallel_func(inner_n_jobs, backend):
    return Parallel(n_jobs=inner_n_jobs, backend=backend)(
        delayed(square)(i) for i in range(3))


###############################################################################
def test_cpu_count():
    assert cpu_count() > 0


def test_effective_n_jobs():
    assert effective_n_jobs() > 0


###############################################################################
# Test parallel

@parametrize('backend', ALL_VALID_BACKENDS)
@parametrize('n_jobs', [1, 2, -1, -2])
@parametrize('verbose', [2, 11, 100])
def test_simple_parallel(backend, n_jobs, verbose):
    assert ([square(x) for x in range(5)] ==
            Parallel(n_jobs=n_jobs, backend=backend,
                     verbose=verbose)(
                delayed(square)(x) for x in range(5)))


@parametrize('backend', ALL_VALID_BACKENDS)
def test_main_thread_renamed_no_warning(backend, monkeypatch):
    # Check that no default backend relies on the name of the main thread:
    # https://github.com/joblib/joblib/issues/180#issuecomment-253266247
    # Some programs use a different name for the main thread. This is the case
    # for uWSGI apps for instance.
    monkeypatch.setattr(target=threading.current_thread(), name='name',
                        value='some_new_name_for_the_main_thread')

    with warns(None) as warninfo:
        results = Parallel(n_jobs=2, backend=backend)(
            delayed(square)(x) for x in range(3))
        assert results == [0, 1, 4]

    # Due to the default parameters of LokyBackend, there is a chance that
    # warninfo catches Warnings from worker timeouts. We remove it if it exists
    warninfo = [w for w in warninfo if "worker timeout" not in str(w.message)]

    # The multiprocessing backend will raise a warning when detecting that is
    # started from the non-main thread. Let's check that there is no false
    # positive because of the name change.
    assert len(warninfo) == 0


def _assert_warning_nested(backend, inner_n_jobs, expected):
    with warns(None) as records:
        parallel_func(backend=backend, inner_n_jobs=inner_n_jobs)

    if expected:
        # with threading, we might see more that one records
        if len(records) > 0:
            return 'backed parallel loops cannot' in records[0].message.args[0]
        return False
    else:
        assert len(records) == 0
        return True


@with_multiprocessing
@parametrize('parent_backend,child_backend,expected', [
    ('loky', 'multiprocessing', True), ('loky', 'loky', False),
    ('multiprocessing', 'multiprocessing', True),
    ('multiprocessing', 'loky', True),
    ('threading', 'multiprocessing', True),
    ('threading', 'loky', True),
])
def test_nested_parallel_warnings(parent_backend, child_backend, expected):

    # no warnings if inner_n_jobs=1
    Parallel(n_jobs=2, backend=parent_backend)(
        delayed(_assert_warning_nested)(
            backend=child_backend, inner_n_jobs=1,
            expected=False)
        for _ in range(5))

    #  warnings if inner_n_jobs != 1 and expected
    res = Parallel(n_jobs=2, backend=parent_backend)(
        delayed(_assert_warning_nested)(
            backend=child_backend, inner_n_jobs=2,
            expected=expected)
        for _ in range(5))

    # warning handling is not thread safe. One thread might see multiple
    # warning or no warning at all.
    if parent_backend == "threading":
        assert any(res)
    else:
        assert all(res)


def nested_loop(backend):
    Parallel(n_jobs=2, backend=backend)(
        delayed(square)(.01) for _ in range(2))


@parametrize('child_backend', BACKENDS)
@parametrize('parent_backend', BACKENDS)
def test_nested_loop(parent_backend, child_backend):
    Parallel(n_jobs=2, backend=parent_backend)(
        delayed(nested_loop)(child_backend) for _ in range(2))


def test_mutate_input_with_threads():
    """Input is mutable when using the threading backend"""
    q = Queue(maxsize=5)
    Parallel(n_jobs=2, backend="threading")(
        delayed(q.put)(1) for _ in range(5))
    assert q.full()


@parametrize('n_jobs', [1, 2, 3])
def test_parallel_kwargs(n_jobs):
    """Check the keyword argument processing of pmap."""
    lst = range(10)
    assert ([f(x, y=1) for x in lst] ==
            Parallel(n_jobs=n_jobs)(delayed(f)(x, y=1) for x in lst))


@parametrize('backend', PARALLEL_BACKENDS)
def test_parallel_as_context_manager(backend):
    lst = range(10)
    expected = [f(x, y=1) for x in lst]

    with Parallel(n_jobs=4, backend=backend) as p:
        # Internally a pool instance has been eagerly created and is managed
        # via the context manager protocol
        managed_backend = p._backend
        if mp is not None:
            assert managed_backend is not None
            assert get_workers(managed_backend) is not None

        # We make call with the managed parallel object several times inside
        # the managed block:
        assert expected == p(delayed(f)(x, y=1) for x in lst)
        assert expected == p(delayed(f)(x, y=1) for x in lst)

        # Those calls have all used the same pool instance:
        if mp is not None:
            assert get_workers(managed_backend) is get_workers(p._backend)

    # As soon as we exit the context manager block, the pool is terminated and
    # no longer referenced from the parallel object:
    if mp is not None:
        assert get_workers(p._backend) is None

    # It's still possible to use the parallel instance in non-managed mode:
    assert expected == p(delayed(f)(x, y=1) for x in lst)
    if mp is not None:
        assert get_workers(p._backend) is None


@with_multiprocessing
def test_parallel_pickling():
    """ Check that pmap captures the errors when it is passed an object
        that cannot be pickled.
    """
    class UnpicklableObject:
        def __reduce__(self):
            raise RuntimeError()

    with raises(RuntimeError):
        Parallel(n_jobs=2)(delayed(id)(UnpicklableObject()) for _ in range(10))


@parametrize('backend', PARALLEL_BACKENDS)
def test_parallel_timeout_success(backend):
    # Check that timeout isn't thrown when function is fast enough
    assert len(Parallel(n_jobs=2, backend=backend, timeout=10)(
        delayed(sleep)(0.001) for x in range(10))) == 10


@with_multiprocessing
@parametrize('backend', PARALLEL_BACKENDS)
def test_parallel_timeout_fail(backend):
    # Check that timeout properly fails when function is too slow
    with raises(TimeoutError):
        Parallel(n_jobs=2, backend=backend, timeout=0.01)(
            delayed(sleep)(10) for x in range(10))


@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_error_capture(backend):
    # Check that error are captured, and that correct exceptions
    # are raised.
    if mp is not None:
        # A JoblibException will be raised only if there is indeed
        # multiprocessing
        with raises(JoblibException):
            Parallel(n_jobs=2, backend=backend)(
                [delayed(division)(x, y)
                    for x, y in zip((0, 1), (1, 0))])
        with raises(WorkerInterrupt):
            Parallel(n_jobs=2, backend=backend)(
                [delayed(interrupt_raiser)(x) for x in (1, 0)])

        # Try again with the context manager API
        with Parallel(n_jobs=2, backend=backend) as parallel:
            assert get_workers(parallel._backend) is not None
            original_workers = get_workers(parallel._backend)

            with raises(JoblibException):
                parallel([delayed(division)(x, y)
                          for x, y in zip((0, 1), (1, 0))])

            # The managed pool should still be available and be in a working
            # state despite the previously raised (and caught) exception
            assert get_workers(parallel._backend) is not None

            # The pool should have been interrupted and restarted:
            assert get_workers(parallel._backend) is not original_workers

            assert ([f(x, y=1) for x in range(10)] ==
                    parallel(delayed(f)(x, y=1) for x in range(10)))

            original_workers = get_workers(parallel._backend)
            with raises(WorkerInterrupt):
                parallel([delayed(interrupt_raiser)(x) for x in (1, 0)])

            # The pool should still be available despite the exception
            assert get_workers(parallel._backend) is not None

            # The pool should have been interrupted and restarted:
            assert get_workers(parallel._backend) is not original_workers

            assert ([f(x, y=1) for x in range(10)] ==
                    parallel(delayed(f)(x, y=1) for x in range(10)))

        # Check that the inner pool has been terminated when exiting the
        # context manager
        assert get_workers(parallel._backend) is None
    else:
        with raises(KeyboardInterrupt):
            Parallel(n_jobs=2)(
                [delayed(interrupt_raiser)(x) for x in (1, 0)])

    # wrapped exceptions should inherit from the class of the original
    # exception to make it easy to catch them
    with raises(ZeroDivisionError):
        Parallel(n_jobs=2)(
            [delayed(division)(x, y) for x, y in zip((0, 1), (1, 0))])

    with raises(MyExceptionWithFinickyInit):
        Parallel(n_jobs=2, verbose=0)(
            (delayed(exception_raiser)(i, custom_exception=True)
             for i in range(30)))

    try:
        # JoblibException wrapping is disabled in sequential mode:
        ex = JoblibException()
        Parallel(n_jobs=1)(
            delayed(division)(x, y) for x, y in zip((0, 1), (1, 0)))
    except Exception as ex:
        assert not isinstance(ex, JoblibException)


def consumer(queue, item):
    queue.append('Consumed %s' % item)


@parametrize('backend', BACKENDS)
@parametrize('batch_size, expected_queue',
             [(1, ['Produced 0', 'Consumed 0',
                   'Produced 1', 'Consumed 1',
                   'Produced 2', 'Consumed 2',
                   'Produced 3', 'Consumed 3',
                   'Produced 4', 'Consumed 4',
                   'Produced 5', 'Consumed 5']),
              (4, [  # First Batch
                  'Produced 0', 'Produced 1', 'Produced 2', 'Produced 3',
                  'Consumed 0', 'Consumed 1', 'Consumed 2', 'Consumed 3',
                     # Second batch
                  'Produced 4', 'Produced 5', 'Consumed 4', 'Consumed 5'])])
def test_dispatch_one_job(backend, batch_size, expected_queue):
    """ Test that with only one job, Parallel does act as a iterator.
    """
    queue = list()

    def producer():
        for i in range(6):
            queue.append('Produced %i' % i)
            yield i

    Parallel(n_jobs=1, batch_size=batch_size, backend=backend)(
        delayed(consumer)(queue, x) for x in producer())
    assert queue == expected_queue
    assert len(queue) == 12


@with_multiprocessing
@parametrize('backend', PARALLEL_BACKENDS)
def test_dispatch_multiprocessing(backend):
    """ Check that using pre_dispatch Parallel does indeed dispatch items
        lazily.
    """
    manager = mp.Manager()
    queue = manager.list()

    def producer():
        for i in range(6):
            queue.append('Produced %i' % i)
            yield i

    Parallel(n_jobs=2, batch_size=1, pre_dispatch=3, backend=backend)(
        delayed(consumer)(queue, 'any') for _ in producer())

    # Only 3 tasks are dispatched out of 6. The 4th task is dispatched only
    # after any of the first 3 jobs have completed.
    first_four = list(queue)[:4]
    # The the first consumption event can sometimes happen before the end of
    # the dispatching, hence, pop it before introspecting the "Produced" events
    first_four.remove('Consumed any')
    assert first_four == ['Produced 0', 'Produced 1', 'Produced 2']
    assert len(queue) == 12


def test_batching_auto_threading():
    # batching='auto' with the threading backend leaves the effective batch
    # size to 1 (no batching) as it has been found to never be beneficial with
    # this low-overhead backend.

    with Parallel(n_jobs=2, batch_size='auto', backend='threading') as p:
        p(delayed(id)(i) for i in range(5000))  # many very fast tasks
        assert p._backend.compute_batch_size() == 1


@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_batching_auto_subprocesses(backend):
    with Parallel(n_jobs=2, batch_size='auto', backend=backend) as p:
        p(delayed(id)(i) for i in range(5000))  # many very fast tasks

        # It should be strictly larger than 1 but as we don't want heisen
        # failures on clogged CI worker environment be safe and only check that
        # it's a strictly positive number.
        assert p._backend.compute_batch_size() > 0


def test_exception_dispatch():
    """Make sure that exception raised during dispatch are indeed captured"""
    with raises(ValueError):
        Parallel(n_jobs=2, pre_dispatch=16, verbose=0)(
            delayed(exception_raiser)(i) for i in range(30))


def test_nested_exception_dispatch():
    """Ensure TransportableException objects for nested joblib cases gets
    propagated."""
    with raises(JoblibException):
        Parallel(n_jobs=2, pre_dispatch=16, verbose=0)(
            delayed(SafeFunction(exception_raiser))(i) for i in range(30))


def _reload_joblib():
    # Retrieve the path of the parallel module in a robust way
    joblib_path = Parallel.__module__.split(os.sep)
    joblib_path = joblib_path[:1]
    joblib_path.append('parallel.py')
    joblib_path = '/'.join(joblib_path)
    module = __import__(joblib_path)
    # Reload the module. This should trigger a fail
    reload(module)


def test_multiple_spawning():
    # Test that attempting to launch a new Python after spawned
    # subprocesses will raise an error, to avoid infinite loops on
    # systems that do not support fork
    if not int(os.environ.get('JOBLIB_MULTIPROCESSING', 1)):
        raise SkipTest()
    with raises(ImportError):
        Parallel(n_jobs=2, pre_dispatch='all')(
            [delayed(_reload_joblib)() for i in range(10)])


class FakeParallelBackend(SequentialBackend):
    """Pretends to run concurrently while running sequentially."""

    def configure(self, n_jobs=1, parallel=None, **backend_args):
        self.n_jobs = self.effective_n_jobs(n_jobs)
        self.parallel = parallel
        return n_jobs

    def effective_n_jobs(self, n_jobs=1):
        if n_jobs < 0:
            n_jobs = max(mp.cpu_count() + 1 + n_jobs, 1)
        return n_jobs


def test_invalid_backend():
    with raises(ValueError):
        Parallel(backend='unit-testing')


@parametrize('backend', ALL_VALID_BACKENDS)
def test_invalid_njobs(backend):
    with raises(ValueError) as excinfo:
        Parallel(n_jobs=0, backend=backend)._initialize_backend()
    assert "n_jobs == 0 in Parallel has no meaning" in str(excinfo.value)


def test_register_parallel_backend():
    try:
        register_parallel_backend("test_backend", FakeParallelBackend)
        assert "test_backend" in BACKENDS
        assert BACKENDS["test_backend"] == FakeParallelBackend
    finally:
        del BACKENDS["test_backend"]


def test_overwrite_default_backend():
    assert _active_backend_type() == DefaultBackend
    try:
        register_parallel_backend("threading", BACKENDS["threading"],
                                  make_default=True)
        assert _active_backend_type() == ThreadingBackend
    finally:
        # Restore the global default manually
        parallel.DEFAULT_BACKEND = DEFAULT_BACKEND
    assert _active_backend_type() == DefaultBackend


def check_backend_context_manager(backend_name):
    with parallel_backend(backend_name, n_jobs=3):
        active_backend, active_n_jobs = parallel.get_active_backend()
        assert active_n_jobs == 3
        assert effective_n_jobs(3) == 3
        p = Parallel()
        assert p.n_jobs == 3
        if backend_name == 'multiprocessing':
            assert type(active_backend) == MultiprocessingBackend
            assert type(p._backend) == MultiprocessingBackend
        elif backend_name == 'loky':
            assert type(active_backend) == LokyBackend
            assert type(p._backend) == LokyBackend
        elif backend_name == 'threading':
            assert type(active_backend) == ThreadingBackend
            assert type(p._backend) == ThreadingBackend
        elif backend_name.startswith('test_'):
            assert type(active_backend) == FakeParallelBackend
            assert type(p._backend) == FakeParallelBackend


all_backends_for_context_manager = PARALLEL_BACKENDS[:]
all_backends_for_context_manager.extend(
    ['test_backend_%d' % i for i in range(3)]
)


@with_multiprocessing
@parametrize('backend', all_backends_for_context_manager)
def test_backend_context_manager(monkeypatch, backend):
    if backend not in BACKENDS:
        monkeypatch.setitem(BACKENDS, backend, FakeParallelBackend)

    assert _active_backend_type() == DefaultBackend
    # check that this possible to switch parallel backends sequentially
    check_backend_context_manager(backend)

    # The default backend is restored
    assert _active_backend_type() == DefaultBackend

    # Check that context manager switching is thread safe:
    Parallel(n_jobs=2, backend='threading')(
        delayed(check_backend_context_manager)(b)
        for b in all_backends_for_context_manager if not b)

    # The default backend is again restored
    assert _active_backend_type() == DefaultBackend


class ParameterizedParallelBackend(SequentialBackend):
    """Pretends to run conncurrently while running sequentially."""

    def __init__(self, param=None):
        if param is None:
            raise ValueError('param should not be None')
        self.param = param


def test_parameterized_backend_context_manager(monkeypatch):
    monkeypatch.setitem(BACKENDS, 'param_backend',
                        ParameterizedParallelBackend)
    assert _active_backend_type() == DefaultBackend

    with parallel_backend('param_backend', param=42, n_jobs=3):
        active_backend, active_n_jobs = parallel.get_active_backend()
        assert type(active_backend) == ParameterizedParallelBackend
        assert active_backend.param == 42
        assert active_n_jobs == 3
        p = Parallel()
        assert p.n_jobs == 3
        assert p._backend is active_backend
        results = p(delayed(sqrt)(i) for i in range(5))
    assert results == [sqrt(i) for i in range(5)]

    # The default backend is again restored
    assert _active_backend_type() == DefaultBackend


def test_direct_parameterized_backend_context_manager():
    assert _active_backend_type() == DefaultBackend

    # Check that it's possible to pass a backend instance directly,
    # without registration
    with parallel_backend(ParameterizedParallelBackend(param=43), n_jobs=5):
        active_backend, active_n_jobs = parallel.get_active_backend()
        assert type(active_backend) == ParameterizedParallelBackend
        assert active_backend.param == 43
        assert active_n_jobs == 5
        p = Parallel()
        assert p.n_jobs == 5
        assert p._backend is active_backend
        results = p(delayed(sqrt)(i) for i in range(5))
    assert results == [sqrt(i) for i in range(5)]

    # The default backend is again restored
    assert _active_backend_type() == DefaultBackend


###############################################################################
# Test helpers
def test_joblib_exception():
    # Smoke-test the custom exception
    e = JoblibException('foobar')
    # Test the repr
    repr(e)
    # Test the pickle
    pickle.dumps(e)


def test_safe_function():
    safe_division = SafeFunction(division)
    with raises(JoblibException):
        safe_division(1, 0)


@parametrize('batch_size', [0, -1, 1.42])
def test_invalid_batch_size(batch_size):
    with raises(ValueError):
        Parallel(batch_size=batch_size)


@parametrize('n_tasks, n_jobs, pre_dispatch, batch_size',
             [(2, 2, 'all', 'auto'),
              (2, 2, 'n_jobs', 'auto'),
              (10, 2, 'n_jobs', 'auto'),
              (517, 2, 'n_jobs', 'auto'),
              (10, 2, 'n_jobs', 'auto'),
              (10, 4, 'n_jobs', 'auto'),
              (200, 12, 'n_jobs', 'auto'),
              (25, 12, '2 * n_jobs', 1),
              (250, 12, 'all', 1),
              (250, 12, '2 * n_jobs', 7),
              (200, 12, '2 * n_jobs', 'auto')])
def test_dispatch_race_condition(n_tasks, n_jobs, pre_dispatch, batch_size):
    # Check that using (async-)dispatch does not yield a race condition on the
    # iterable generator that is not thread-safe natively.
    # This is a non-regression test for the "Pool seems closed" class of error
    params = {'n_jobs': n_jobs, 'pre_dispatch': pre_dispatch,
              'batch_size': batch_size}
    expected = [square(i) for i in range(n_tasks)]
    results = Parallel(**params)(delayed(square)(i) for i in range(n_tasks))
    assert results == expected


@with_multiprocessing
def test_default_mp_context():
    p = Parallel(n_jobs=2, backend='multiprocessing')
    context = p._backend_args.get('context')
    if sys.version_info >= (3, 4):
        start_method = context.get_start_method()
        # Under Python 3.4+ the multiprocessing context can be configured
        # by an environment variable
        env_method = os.environ.get('JOBLIB_START_METHOD', '').strip() or None
        if env_method is None:
            # Check the default behavior
            if sys.platform == 'win32':
                assert start_method == 'spawn'
            else:
                assert start_method == 'fork'
        else:
            assert start_method == env_method
    else:
        assert context is None


@with_numpy
@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_no_blas_crash_or_freeze_with_subprocesses(backend):
    if backend == 'multiprocessing':
        if sys.version_info < (3, 4):
            raise SkipTest('multiprocessing can cause BLAS freeze on old '
                           'Python that relies on fork.')
        # Use the spawn backend that is both robust and available on all
        # platforms
        backend = mp.get_context('spawn')

    # Check that on recent Python version, the 'spawn' start method can make
    # it possible to use multiprocessing in conjunction of any BLAS
    # implementation that happens to be used by numpy with causing a freeze or
    # a crash
    rng = np.random.RandomState(42)

    # call BLAS DGEMM to force the initialization of the internal thread-pool
    # in the main process
    a = rng.randn(1000, 1000)
    np.dot(a, a.T)

    # check that the internal BLAS thread-pool is not in an inconsistent state
    # in the worker processes managed by multiprocessing
    Parallel(n_jobs=2, backend=backend)(
        delayed(np.dot)(a, a.T) for i in range(2))


CUSTOM_BACKEND_SCRIPT_TEMPLATE = """\
from joblib import Parallel, delayed

def square(x):
    return x**2

backend = "{}"
if backend == "spawn":
    from multiprocessing import get_context
    backend = get_context(backend)

print(Parallel(n_jobs=2, backend=backend)(
        delayed(square)(i) for i in range(5)))
"""


@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS +
             ([] if sys.version_info[:2] < (3, 4) or mp is None
              else ['spawn']))
def test_parallel_with_interactively_defined_functions(backend):
    # When using the "-c" flag, interactive functions defined in __main__
    # should work with any backend.
    code = CUSTOM_BACKEND_SCRIPT_TEMPLATE.format(backend)
    check_subprocess_call([sys.executable, '-c', code],
                          stdout_regex=r'\[0, 1, 4, 9, 16\]',
                          timeout=2)


DEFAULT_BACKEND_SCRIPT_CONTENT = """\
import sys
# Make sure that joblib is importable in the subprocess launching this
# script. This is needed in case we run the tests from the joblib root
# folder without having installed joblib
sys.path.insert(0, {joblib_root_folder!r})

from joblib import Parallel, delayed


def square(x):
    return x ** 2
print(Parallel(n_jobs=2)(delayed(square)(i) for i in range(5)))
""".format(joblib_root_folder=os.path.dirname(
    os.path.dirname(joblib.__file__)))


@with_multiprocessing
def test_parallel_with_interactively_defined_functions_default_backend(tmpdir):
    # The default backend (loky) accepts interactive functions defined in
    # __main__ and does not require if __name__ == '__main__' even when
    # the __main__ module is defined by the result of the execution of a
    # filesystem script.
    script = tmpdir.join('joblib_default_backend_script.py')
    script.write(DEFAULT_BACKEND_SCRIPT_CONTENT)
    check_subprocess_call([sys.executable, script.strpath],
                          stdout_regex=r'\[0, 1, 4, 9, 16\]',
                          timeout=2)


def test_parallel_with_exhausted_iterator():
    exhausted_iterator = iter([])
    assert Parallel(n_jobs=2)(exhausted_iterator) == []


def check_memmap(a):
    if not isinstance(a, np.memmap):
        raise TypeError('Expected np.memmap instance, got %r',
                        type(a))
    return a.copy()  # return a regular array instead of a memmap


@with_numpy
@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_auto_memmap_on_arrays_from_generator(backend):
    # Non-regression test for a problem with a bad interaction between the
    # GC collecting arrays recently created during iteration inside the
    # parallel dispatch loop and the auto-memmap feature of Parallel.
    # See: https://github.com/joblib/joblib/pull/294
    def generate_arrays(n):
        for i in range(n):
            yield np.ones(10, dtype=np.float32) * i
    # Use max_nbytes=1 to force the use of memory-mapping even for small
    # arrays
    results = Parallel(n_jobs=2, max_nbytes=1, backend=backend)(
        delayed(check_memmap)(a) for a in generate_arrays(100))
    for result, expected in zip(results, generate_arrays(len(results))):
        np.testing.assert_array_equal(expected, result)


def identity(arg):
    return arg


@with_numpy
@with_multiprocessing
def test_memmap_with_big_offset(tmpdir):
    fname = tmpdir.join('test.mmap').strpath
    size = mmap.ALLOCATIONGRANULARITY
    obj = [np.zeros(size, dtype='uint8'), np.ones(size, dtype='uint8')]
    dump(obj, fname)
    memmap = load(fname, mmap_mode='r')
    result, = Parallel(n_jobs=2)(delayed(identity)(memmap) for _ in [0])
    assert isinstance(memmap[1], np.memmap)
    assert memmap[1].offset > size
    np.testing.assert_array_equal(obj, result)


def test_warning_about_timeout_not_supported_by_backend():
    with warns(None) as warninfo:
        Parallel(timeout=1)(delayed(square)(i) for i in range(50))
    assert len(warninfo) == 1
    w = warninfo[0]
    assert isinstance(w.message, UserWarning)
    assert str(w.message) == (
        "The backend class 'SequentialBackend' does not support timeout. "
        "You have set 'timeout=1' in Parallel but the 'timeout' parameter "
        "will not be used.")


@parametrize('backend', ALL_VALID_BACKENDS)
@parametrize('n_jobs', [1, 2, -2, -1])
def test_abort_backend(n_jobs, backend):
    delays = ["a"] + [10] * 100

    if os.environ.get("TRAVIS_OS_NAME") is not None and n_jobs < 0:
        # Use only up to 8 cpu in travis as cpu_count return 32 whereas we
        # only access 2 cores.
        n_jobs += 8

    with raises(TypeError):
        t_start = time.time()
        Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(time.sleep)(i) for i in delays)
    dt = time.time() - t_start
    assert dt < 3


@with_numpy
@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_memmapping_leaks(backend, tmpdir):
    # Non-regression test for memmapping backends. Ensure that the data
    # does not stay too long in memory
    tmpdir = tmpdir.strpath

    # Use max_nbytes=1 to force the use of memory-mapping even for small
    # arrays
    with Parallel(n_jobs=2, max_nbytes=1, backend=backend,
                  temp_folder=tmpdir) as p:
        p(delayed(check_memmap)(a) for a in [np.random.random(10)] * 2)

        # The memmap folder should not be clean in the context scope
        assert len(os.listdir(tmpdir)) > 0

    # Make sure that the shared memory is cleaned at the end when we exit
    # the context
    assert not os.listdir(tmpdir)

    # Make sure that the shared memory is cleaned at the end of a call
    p = Parallel(n_jobs=2, max_nbytes=1, backend=backend)
    p(delayed(check_memmap)(a) for a in [np.random.random(10)] * 2)

    assert not os.listdir(tmpdir)


def test_lambda_expression():
    # cloudpickle is used to pickle delayed callables
    for backend in ALL_VALID_BACKENDS:
        results = Parallel(n_jobs=2, backend=backend)(
            delayed(lambda x: x ** 2)(i) for i in range(10))
        assert results == [i ** 2 for i in range(10)]


def test_delayed_check_pickle_deprecated():

    class UnpicklableCallable(object):

        def __call__(self, *args, **kwargs):
            return 42

        def __reduce__(self):
            raise ValueError()

    with warns(DeprecationWarning):
        f, args, kwargs = delayed(lambda x: 42, check_pickle=False)('a')
    assert f('a') == 42
    assert args == ('a',)
    assert kwargs == dict()

    with warns(DeprecationWarning):
        f, args, kwargs = delayed(UnpicklableCallable(),
                                  check_pickle=False)('a', option='b')
        assert f('a', option='b') == 42
        assert args == ('a',)
        assert kwargs == dict(option='b')

    with warns(DeprecationWarning):
        with raises(ValueError):
            delayed(UnpicklableCallable(), check_pickle=True)


@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky'])
def test_backend_batch_statistics_reset(backend):
    """Test that a parallel backend correctly resets its batch statistics."""
    relative_tolerance = 0.2
    n_jobs = 2
    n_inputs = 500
    task_time = 2. / n_inputs

    p = Parallel(verbose=10, n_jobs=n_jobs, backend=backend)
    start_time = time.time()
    p(delayed(time.sleep)(task_time) for i in range(n_inputs))
    ref_time = time.time() - start_time
    assert (p._backend._effective_batch_size ==
            p._backend._DEFAULT_EFFECTIVE_BATCH_SIZE)
    assert (p._backend._smoothed_batch_duration ==
            p._backend._DEFAULT_SMOOTHED_BATCH_DURATION)

    start_time = time.time()
    p(delayed(time.sleep)(task_time) for i in range(n_inputs))
    test_time = time.time() - start_time
    assert (p._backend._effective_batch_size ==
            p._backend._DEFAULT_EFFECTIVE_BATCH_SIZE)
    assert (p._backend._smoothed_batch_duration ==
            p._backend._DEFAULT_SMOOTHED_BATCH_DURATION)

    # Tolerance in the timing comparison to avoid random failures on CIs
    assert test_time / ref_time <= 1 + relative_tolerance