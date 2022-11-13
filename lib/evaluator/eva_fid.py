import torch
import numpy as np
import re
import urllib
import requests
import os
import os.path as osp
import hashlib
import html
import glob
import io
import uuid
import scipy.linalg

from typing import Any, List, Tuple, Union

from ..log_service import print_log, torch_to_numpy

from .eva_base import base_evaluator, register

detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'

def make_cache_dir_path(*paths: str) -> str:
    return os.path.join('.cache', *paths)

def is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    """Determine whether the given object is a valid URL string."""
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True

def open_url(url: str, 
             cache_dir: str = None, 
             num_attempts: int = 10, 
             verbose: bool = True, 
             return_filename: bool = False, 
             cache: bool = True) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    assert is_url(url)

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file) # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)

def get_feature_detector(url, 
                         device=torch.device('cpu'), 
                         num_gpus=1, 
                         rank=0, 
                         verbose=False):
    assert 0 <= rank < num_gpus
    is_leader = (rank == 0)
    if not is_leader and num_gpus > 1:
        torch.distributed.barrier() # leader goes first
    with open_url(url, verbose=(verbose and is_leader)) as f:
        detector = torch.jit.load(f).eval().to(device)
    if is_leader and num_gpus > 1:
        torch.distributed.barrier() # others follow
    return detector

@register('fid')
class fid_evaluator(base_evaluator):
    def __init__(self, 
                 device='cpu',
                 sample_real_n=None, 
                 sample_fake_n=None, 
                 dsstat_cachefile_tag=None, **dummy):
        super().__init__()
        self.symbol = 'fid'
        self.data_fake_feat = None
        self.data_real_feat = None
        self.data_fn = None
        self.sample_real_n = sample_real_n
        self.sample_fake_n = sample_fake_n
        self.dsstat_use_cache = False 
        self.dsstat_cache_file = None
        self.device = torch.device(device+':{}'.format(self.rank)) if device=='cuda' else device
        if dsstat_cachefile_tag is not None:
            self.dsstat_cache_file = osp.join(
                '.cache', dsstat_cachefile_tag+'_real_feat.npy')
            self.dsstat_use_cache = osp.isfile(self.dsstat_cache_file)
        if self.rank == 0:
            if self.dsstat_use_cache:
                print_log('Load dsstat_cache from {}'.format(self.dsstat_cache_file))
            elif self.dsstat_cache_file is None:
                print_log('Do not use dsstat_cache.')
            else:
                print_log('Do not find dsstat_cache {}, plan to write one.'.format(self.dsstat_cache_file))
        self.detector = get_feature_detector(
            detector_url,
            device=self.device,
            num_gpus=self.world_size,
            rank=self.rank)

    def add_batch(self, 
                  fake, 
                  real, 
                  fn=None,
                  **dummy):

        if isinstance(real, torch.Tensor):
            fake, real = fake.to(self.device), real.to(self.device)
        else:
            fake = torch.Tensor(fake).to(self.device)
            real = torch.Tensor(real).to(self.device)

        fake_feat = self.detector(fake, return_features=True)
        fake_feat = torch_to_numpy(fake_feat).astype(float) # make if float64
        real_feat = None
        if not self.dsstat_use_cache:
            real_feat = self.detector(real, return_features=True)
            real_feat = torch_to_numpy(real_feat).astype(float) # make if float64
            
        if not self.dsstat_use_cache:
            sync_feat_fn = self.sync([fake_feat, real_feat, fn])
            fake_feat, real_feat, fn = zip(*sync_feat_fn)
            fake_feat = self.zipzap_arrange(fake_feat)
            real_feat = self.zipzap_arrange(real_feat)
            fn = self.zipzap_arrange(fn)
        else:
            sync_feat_fn = self.sync([fake_feat, fn])
            fake_feat, fn = zip(*sync_feat_fn)
            fake_feat = self.zipzap_arrange(fake_feat)
            fn = self.zipzap_arrange(fn)

        if self.data_fn is None:
            self.data_fn = fn
        else:
            self.data_fn += fn

        if self.data_fake_feat is None:
            self.data_fake_feat = [fake_feat]
        else:
            self.data_fake_feat += [fake_feat]

        if not self.dsstat_use_cache:
            if self.data_real_feat is None:
                self.data_real_feat = [real_feat]
            else:
                self.data_real_feat += [real_feat]

    def compute(self):
        if self.rank == 0:
            fid = self.compute_fid()
        else:
            fid = 0

        sync_fid = self.sync(np.array([fid]).astype(float))
        if self.rank != 0:
            self.final['fid'] = sync_fid[0][0]
        return sync_fid[0][0]

    def compute_fid(self):
        sample_fake_n = self.sample_n if self.sample_fake_n is None else self.sample_fake_n
        sample_real_n = self.sample_n if self.sample_real_n is None else self.sample_real_n
        fake_feat = np.concatenate(self.data_fake_feat, axis=0)[0:sample_fake_n]
        if self.dsstat_use_cache:
            real_feat = np.load(self.dsstat_cache_file)[0:sample_real_n]
        else:
            real_feat = np.concatenate(self.data_real_feat, axis=0)[0:sample_real_n]

        if (self.dsstat_cache_file is not None) and (not self.dsstat_use_cache):
            # save the cache
            if not osp.isdir(osp.dirname(self.dsstat_cache_file)):
                os.makedirs(osp.dirname(self.dsstat_cache_file))
            print_log('Save dsstat_cache to {}'.format(self.dsstat_cache_file))
            np.save(self.dsstat_cache_file, real_feat)

        mu_fake = fake_feat.mean(0)
        sigma_fake = (fake_feat.T @ fake_feat)/sample_fake_n - np.outer(mu_fake, mu_fake)
        mu_real = real_feat.mean(0)
        sigma_real = (real_feat.T @ real_feat)/sample_real_n - np.outer(mu_real, mu_real)
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_fake + sigma_real - s * 2))

        self.final['fid'] = fid
        return fid

    def one_line_summary(self):
        print_log('Evaluator fid: {:.4f}'.format(self.final['fid']))

    def clear_data(self):
        self.data_real_feat = None
        self.data_fake_feat = None
        self.data_fn = None
