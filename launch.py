from functools import partial
import os
import signal
import torch.distributed.elastic.rendezvous.static_tcp_rendezvous as rendezvous
import torch.distributed.elastic.multiprocessing.api as multiprocessing_api
from torch.distributed.elastic.multiprocessing.subprocess_handler.subprocess_handler import SubprocessHandler

os.environ['USE_LIBUV'] = '0'
rendezvous.TCPStore = partial(rendezvous.TCPStore, use_libuv=False)
_original_close = SubprocessHandler.close


def close_worker(self, death_sig=None):
    return _original_close(self, death_sig=signal.SIGTERM)


if os.name == 'nt':
    os.environ['TORCHELASTIC_SIGNALS_TO_HANDLE'] = 'SIGINT,SIGTERM'
    SubprocessHandler.close = close_worker
    multiprocessing_api._get_default_signal = lambda: signal.SIGTERM
    multiprocessing_api._get_kill_signal = lambda: signal.SIGTERM

if __name__ == '__main__':
    from torch.distributed.run import main
    main()
