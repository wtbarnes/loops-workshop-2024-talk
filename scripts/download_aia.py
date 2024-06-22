import time

import astropy.time
import astropy.units as u
import click
import parfive

from astropy.coordinates import SkyCoord
from sunpy.coordinates import Helioprojective, get_horizons_coord
from sunpy.net import attrs, jsoc
from sunpy.net.attr import and_
from sunpy.time import parse_time


@click.command()
@click.option('--email', type=str)
@click.option('--channels', type=str, default=None)
@click.option('--sample', type=(float, str))
@click.option('--download-dir', type=str, default=None)
@click.option('--start-time', type=str)
@click.option('--interval', type=(float, str))
@click.option('--cutout', type=(float, float, float, float, str), default=None)
@click.option('--no-export', is_flag=True, default=False, help='If set, skip the export/download steps.')
def cli(email,
        channels,
        sample,
        download_dir,
        start_time,
        interval,
        cutout,
        no_export):
    """
    Download AIA data from the JSOC. This downloads the files as tar balls
    which is much quicker than downloading the files individually.
    """
    channels = '94,131,171,193,211,335' if channels is None else channels
    interval = u.Quantity(*interval)
    sample = u.Quantity(*sample)
    start_time = parse_time(start_time)
    time_range = attrs.Time(start_time, end=start_time+interval)

    search_attrs = (
        time_range &
        attrs.Sample(sample) &
        attrs.jsoc.PrimeKey('WAVELNTH', channels) &
        attrs.jsoc.Series.aia_lev1_euv_12s & 
        attrs.jsoc.Notify(email) & 
        attrs.jsoc.Segment.image &
        attrs.jsoc.Protocol('fits')
    )
    if cutout:
        blc_Tx, blc_Ty, trc_Tx, trc_Ty, obstime = cutout
        obstime = parse_time(obstime)
        observer = get_horizons_coord('SDO', time=obstime)
        frame = Helioprojective(obstime=obstime, observer=observer)
        blc = SkyCoord(Tx=blc_Tx, Ty=blc_Ty, unit='arcsec', frame=frame)
        trc = SkyCoord(Tx=trc_Tx, Ty=trc_Ty, unit='arcsec', frame=frame)
        cutout_attr = attrs.jsoc.Cutout(blc, top_right=trc, tracking=True)
        search_attrs = search_attrs & cutout_attr
    jsoc_client = jsoc.JSOCClient()
    q = jsoc_client.search(search_attrs)
    click.echo(q)
    if not no_export:
        exp_req = jsoc_client.request_data(q, method='url-tar')
        dl = parfive.Downloader(max_conn=1, max_splits=1)
        click.echo(exp_req)
        exp_req.wait(sleep=60, verbose=True, retries_notfound=10)
        if exp_req.has_succeeded():
            if download_dir:
                for url in exp_req.urls['url']:
                    dl.enqueue_file(url, path=download_dir)
        if download_dir:
            files = dl.download()
            click.echo(files)
            click.echo('Retrying if there were errors')
            files = dl.retry(files)  # Retry if there were any errors
            click.echo(files)


if __name__ == '__main__':
    cli()
